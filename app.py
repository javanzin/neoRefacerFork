import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gradio as gr
from refacer import Refacer
import argparse
import ngrok
import imageio
import cv2
import numpy as np
from PIL import Image
import tempfile
import base64
import pyfiglet
import shutil
import time
import ffmpeg
from urllib.parse import quote
from gradio.processing_utils import save_file_to_cache
from gradio.utils import get_upload_folder

print("\033[94m" + pyfiglet.Figlet(font='slant').renderText("NeoRefacer") + "\033[0m")

# Video processing history
video_history = []

def _resolve_history_path(value):
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        for item in value:
            resolved = _resolve_history_path(item)
            if resolved:
                return resolved
        return None

    if isinstance(value, dict):
        for key in ("path", "name", "filepath", "file", "image"):
            if key in value:
                resolved = _resolve_history_path(value[key])
                if resolved:
                    return resolved
        return None

    if isinstance(value, str):
        return value

    return str(value)

def _gradio_file_url(path):
    """Build the exact same '/gradio_api/file=...' URL the gr.Video preview player uses.

    gr.Video copies its output into GRADIO_CACHE before serving it, and the file
    endpoint only allows paths inside that cache (or explicit allowed_paths).
    A link built straight from the raw output/ path 403s and Chrome shows it as
    a corrupted video, so we run the file through the same cache step here.
    """
    cached_path = save_file_to_cache(path, cache_dir=get_upload_folder())
    normalized_path = os.path.abspath(cached_path).replace("\\", "/")
    return f"/gradio_api/file={quote(normalized_path, safe='/:._-')}"

def _history_file_link_with_label(value, label):
    path = _resolve_history_path(value)
    if not path or not os.path.exists(path):
        return None

    url = _gradio_file_url(path)
    return f'<a href="{url}" target="_blank" rel="noreferrer">{label}</a>'

def get_video_history():
    """Format video history for display"""
    if not video_history:
        return ""
    
    rows = []
    for entry in video_history:
        output_video = _resolve_history_path(entry.get('output_video'))
        output_label = os.path.basename(output_video) if output_video else 'N/A'
        output_link = _history_file_link_with_label(output_video, output_label) or 'N/A'

        rows.append(
            f"<tr><td style='padding:6px; border-bottom:1px solid #eee;'>{time.strftime('%H:%M:%S', time.localtime(entry['timestamp']))}</td>"
            f"<td style='padding:6px; border-bottom:1px solid #eee;'>{output_link}</td></tr>"
        )

    return (
        "<div style='overflow-x:auto;'>"
        "<table style='width:100%; border-collapse:collapse;'>"
        "<thead><tr><th style='text-align:left; border-bottom:1px solid #ddd; padding:6px;'>Time</th>"
        "<th style='text-align:left; border-bottom:1px solid #ddd; padding:6px;'>Video Link</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )

def clear_video_history():
    """Clear video history"""
    video_history.clear()
    return ""

def cleanup_temp(folder_path):
    try:
        shutil.rmtree(folder_path)
        print("Gradio cache cleared successfully.")
    except Exception as e:
        print(f"Error: {e}")

# Prepare temp folder
os.environ["GRADIO_TEMP_DIR"] = "./tmp"
if os.path.exists("./tmp"):
    cleanup_temp(os.environ['GRADIO_TEMP_DIR'])
if not os.path.exists("./tmp"):
    os.makedirs("./tmp")

# Parse arguments
parser = argparse.ArgumentParser(description='Refacer')
parser.add_argument("--max_num_faces", type=int, default=8)
parser.add_argument("--force_cpu", default=False, action="store_true")
parser.add_argument("--share_gradio", default=False, action="store_true")
parser.add_argument("--server_name", type=str, default="127.0.0.1")
parser.add_argument("--server_port", type=int, default=7860)
parser.add_argument("--colab_performance", default=False, action="store_true")
parser.add_argument("--ngrok", type=str, default=None)
parser.add_argument("--ngrok_region", type=str, default="us")
args = parser.parse_args()

# Initialize
refacer = Refacer(force_cpu=args.force_cpu, colab_performance=args.colab_performance)
num_faces = args.max_num_faces

def create_dummy_image():
    dummy = Image.new('RGB', (1, 1), color=(255, 255, 255))
    temp_file = tempfile.NamedTemporaryFile(delete=False, dir="./tmp", suffix=".png")
    dummy.save(temp_file.name)
    return temp_file.name

def load_first_frame(filepath):
    if filepath is None:
        return None
    frames = imageio.get_reader(filepath)
    return frames.get_data(0)

def load_face_image(value):
    if value is None:
        return None

    if isinstance(value, (list, tuple)):
        for item in value:
            image = load_face_image(item)
            if image is not None:
                return image
        return None

    if isinstance(value, dict):
        for key in ("image", "path", "name", "filepath", "file"):
            if key in value:
                image = load_face_image(value[key])
                if image is not None:
                    return image
        return None

    if isinstance(value, np.ndarray):
        if value.ndim == 2:
            return cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)
        if value.ndim == 3 and value.shape[2] == 4:
            return cv2.cvtColor(value, cv2.COLOR_RGBA2BGR)
        return value

    if hasattr(value, "convert"):
        return cv2.cvtColor(np.array(value.convert("RGB")), cv2.COLOR_RGB2BGR)

    if isinstance(value, str) and os.path.exists(value):
        return cv2.imread(value)

    return None

def extract_faces_auto(filepath, refacer_instance, max_faces=5, isvideo=False):
    if filepath is None:
        return [None] * max_faces

    if isvideo and os.path.getsize(filepath) > 5 * 1024 * 1024:
        print("Video too large for auto-extract, skipping face extraction.")
        return [None] * max_faces

    frame = load_first_frame(filepath)
    if frame is None:
        return [None] * max_faces

    while len(frame.shape) > 3:
        frame = frame[0]

    if frame.shape[-1] != 3:
        raise ValueError(f"Expected last dimension to be 3 (RGB), but got {frame.shape[-1]}")

    temp_image_path = os.path.join("./tmp", f"temp_face_extract_{int(time.time() * 1000)}.png")
    Image.fromarray(frame).save(temp_image_path)

    try:
        faces = refacer_instance.extract_faces_from_image(temp_image_path, max_faces=max_faces)
        return faces + [None] * (max_faces - len(faces))
    finally:
        if os.path.exists(temp_image_path):
            try:
                os.remove(temp_image_path)
            except Exception as e:
                print(f"Warning: Could not delete temp file {temp_image_path}: {e}")

def run_image(*vars):
    image_path = vars[0]
    origins = vars[1:(num_faces+1)]
    destinations = vars[(num_faces+1):(num_faces*2)+1]
    thresholds = vars[(num_faces*2)+1:-2]
    face_mode = vars[-2]
    partial_reface_ratio = vars[-1]

    disable_similarity = (face_mode in ["Single Face", "Multiple Faces"])
    multiple_faces_mode = (face_mode == "Multiple Faces")

    faces = []
    for k in range(num_faces):
        destination_image = load_face_image(destinations[k])
        origin_image = load_face_image(origins[k]) if not multiple_faces_mode else None

        if destination_image is not None:
            faces.append({
                'origin': origin_image,
                'destination': destination_image,
                'threshold': thresholds[k] if not multiple_faces_mode else 0.0
            })

    return refacer.reface_image(image_path, faces, disable_similarity=disable_similarity, multiple_faces_mode=multiple_faces_mode, partial_reface_ratio=partial_reface_ratio)

def _build_video_face_jobs(origins, destinations, thresholds, multiple_faces_mode):
    """Build one job per destination face, keeping Multiple Faces as a single combined job."""
    if multiple_faces_mode:
        # All destination faces are swapped together, by position, in one video.
        faces = []
        labels = []
        for k in range(num_faces):
            dest_files = destinations[k]
            if dest_files is None or (isinstance(dest_files, list) and len(dest_files) == 0):
                continue
            if not isinstance(dest_files, list):
                dest_files = [dest_files]
            for dest_file in dest_files:
                destination_image = load_face_image(dest_file)
                if destination_image is None:
                    continue
                faces.append({'origin': None, 'destination': destination_image, 'threshold': 0.0})
                label = _resolve_history_path(dest_file)
                if label:
                    labels.append(label)

        if not faces:
            return []
        return [{'faces': faces, 'label': ', '.join(labels) if labels else 'Multiple faces'}]

    # Single Face / Faces By Match: one independent job per destination face.
    jobs = []
    for k in range(num_faces):
        dest_files = destinations[k]

        if dest_files is None or (isinstance(dest_files, list) and len(dest_files) == 0):
            continue

        origin_image = load_face_image(origins[k])

        if not isinstance(dest_files, list):
            dest_files = [dest_files]

        for dest_file in dest_files:
            destination_image = load_face_image(dest_file)

            if destination_image is None:
                continue

            label = _resolve_history_path(dest_file)
            jobs.append({
                'faces': [{
                    'origin': origin_image,
                    'destination': destination_image,
                    'threshold': thresholds[k]
                }],
                'label': label or f'Face #{k + 1}'
            })

    return jobs

def run(*vars):
    video_path = vars[0]
    origins = vars[1:(num_faces+1)]
    destinations = vars[(num_faces+1):(num_faces*2)+1]
    thresholds = vars[(num_faces*2)+1:-4]
    preview = vars[-4]
    face_mode = vars[-3]
    partial_reface_ratio = vars[-2]
    use_cache = vars[-1]

    disable_similarity = (face_mode in ["Single Face", "Multiple Faces"])
    multiple_faces_mode = (face_mode == "Multiple Faces")

    jobs = _build_video_face_jobs(origins, destinations, thresholds, multiple_faces_mode)

    if not jobs:
        yield None, None
        return

    last_mp4_path, last_gif_path = None, None

    # When the same target video will be processed for more than one face job,
    # decode + detect it once into a local variable and hand that to every
    # reface() call, instead of relying on refacer's hash-keyed RAM caches.
    precomputed = None
    if use_cache and len(jobs) > 1:
        precomputed = refacer.analyze_video_in_memory(video_path, preview=preview)

    for job in jobs:
        mp4_path, gif_path = refacer.reface(
            video_path,
            job['faces'],
            preview=preview,
            disable_similarity=disable_similarity,
            multiple_faces_mode=multiple_faces_mode,
            partial_reface_ratio=partial_reface_ratio,
            use_cache=use_cache,
            precomputed=precomputed
        )

        if mp4_path:
            # Record the exact same path Gradio serves for the preview player,
            # so the history link always matches a video that actually opens.
            video_history.append({
                'timestamp': int(time.time()),
                'input_video': video_path,
                'output_video': mp4_path,
                'destination_face': job['label']
            })

        last_mp4_path, last_gif_path = mp4_path, gif_path
        yield last_mp4_path, last_gif_path

def toggle_tabs_and_faces(mode, face_tabs, origin_faces):
    if mode == "Single Face":
        tab_updates = [gr.update(visible=(i == 0)) for i in range(len(face_tabs))]
        origin_updates = [gr.update(visible=False) for _ in range(len(origin_faces))]
    elif mode == "Multiple Faces":
        tab_updates = [gr.update(visible=True) for _ in range(len(face_tabs))]
        origin_updates = [gr.update(visible=False) for _ in range(len(origin_faces))]
    else:
        tab_updates = [gr.update(visible=True) for _ in range(len(face_tabs))]
        origin_updates = [gr.update(visible=True) for _ in range(len(origin_faces))]
    return tab_updates + origin_updates
    
def handle_tif_preview(filepath):
    if filepath is None:
        return None
    preview_path = os.path.join("./tmp", f"tif_preview_{int(time.time() * 1000)}.jpg")
    Image.open(filepath).convert('RGB').save(preview_path)
    return preview_path

def rotate_video(video_path, direction):
    if video_path is None:
        return video_path

    rotation_filters = {
        'left': "transpose=2",
        'right': "transpose=1"
    }

    if direction not in rotation_filters:
        return video_path

    output_path = os.path.join("./tmp", f"rotated_{direction}_{int(time.time() * 1000)}.mp4")

    try:
        probe = ffmpeg.probe(video_path)
        has_audio = any(stream.get('codec_type') == 'audio' for stream in probe.get('streams', []))
        input_stream = ffmpeg.input(video_path)
        rotated_video = input_stream.video.filter('transpose', 2 if direction == 'left' else 1)

        if has_audio:
            output = ffmpeg.output(rotated_video, input_stream.audio, output_path, vcodec='libx264', acodec='copy', movflags='+faststart')
        else:
            output = ffmpeg.output(rotated_video, output_path, vcodec='libx264', movflags='+faststart')

        (
            output
            .overwrite_output()
            .run(quiet=True)
        )
        return output_path
    except Exception as e:
        print(f"Error rotating video: {e}")
        return video_path

# --- UI ---
theme = gr.themes.Base(primary_hue="blue", secondary_hue="cyan")

with gr.Blocks(theme=theme, title="NeoRefacer - AI Refacer") as demo:
    with open("icon.png", "rb") as f:
        icon_data = base64.b64encode(f.read()).decode()
    icon_html = f'<img src="data:image/png;base64,{icon_data}" style="width:40px;height:40px;margin-right:10px;">'

    with gr.Row():
        gr.Markdown(f"""
        <div style="display: flex; align-items: center;">
        {icon_html}
        <span style="font-size: 2em; font-weight: bold; color:#2563eb;">NeoRefacer</span>
        </div>
        """)

    # --- IMAGE MODE ---
    with gr.Tab("Image Mode"):
        with gr.Row():
            image_input = gr.Image(label="Original image", type="filepath")
            image_output = gr.Image(label="Refaced image", interactive=False, type="filepath")

        with gr.Row():
            face_mode_image = gr.Radio(["Single Face", "Multiple Faces", "Faces By Match"], value="Single Face", label="Replacement Mode")
            partial_reface_ratio_image = gr.Slider(label="Reface Ratio (0 = Full Face, 0.5 = Half Face)", minimum=0.0, maximum=0.5, value=0.0, step=0.1)
            image_btn = gr.Button("Reface Image", variant="primary")

        origin_image, destination_image, thresholds_image, face_tabs_image = [], [], [], []

        for i in range(num_faces):
            with gr.Tab(f"Face #{i+1}") as tab:
                with gr.Row():
                    origin = gr.Image(label="Face to replace")
                    destination = gr.Image(label="Destination face")
                threshold = gr.Slider(label="Threshold", minimum=0.0, maximum=1.0, value=0.2)
            origin_image.append(origin)
            destination_image.append(destination)
            thresholds_image.append(threshold)
            face_tabs_image.append(tab)

        face_mode_image.change(fn=lambda mode: toggle_tabs_and_faces(mode, face_tabs_image, origin_image), inputs=[face_mode_image], outputs=face_tabs_image + origin_image)
        demo.load(fn=lambda: toggle_tabs_and_faces("Single Face", face_tabs_image, origin_image), inputs=None, outputs=face_tabs_image + origin_image)

        image_btn.click(fn=run_image, inputs=[image_input] + origin_image + destination_image + thresholds_image + [face_mode_image, partial_reface_ratio_image], outputs=[image_output])
        image_input.change(fn=lambda filepath: extract_faces_auto(filepath, refacer, max_faces=num_faces), inputs=image_input, outputs=origin_image)
        image_input.change(fn=lambda _: 0.0, inputs=image_input, outputs=partial_reface_ratio_image)

    # --- GIF MODE ---
    with gr.Tab("GIF Mode"):
        with gr.Row():
            gif_input = gr.File(label="Original GIF", file_types=[".gif"])
            gif_preview = gr.Video(label="GIF Preview", interactive=False)
            gif_output = gr.Video(label="Refaced GIF (MP4)", interactive=False, format="mp4")
            gif_file_output = gr.Image(label="Refaced GIF (GIF)", type="filepath")

        with gr.Row():
            face_mode_gif = gr.Radio(["Single Face", "Multiple Faces", "Faces By Match"], value="Single Face", label="Replacement Mode")
            partial_reface_ratio_gif = gr.Slider(label="Reface Ratio (0 = Full Face, 0.5 = Half Face)", minimum=0.0, maximum=0.5, value=0.0, step=0.1)
            gif_btn = gr.Button("Reface GIF", variant="primary")
            with gr.Column():
                preview_checkbox_gif = gr.Checkbox(label="Preview Generation (skip 90% of frames)", value=False)
                use_cache_gif = gr.Checkbox(label="Enable Cache (Faster subsequent runs)", value=False)

        origin_gif, destination_gif, thresholds_gif, face_tabs_gif = [], [], [], []

        for i in range(num_faces):
            with gr.Tab(f"Face #{i+1}") as tab:
                with gr.Row():
                    origin = gr.Image(label="Face to replace")
                    destination = gr.Image(label="Destination face")
                threshold = gr.Slider(label="Threshold", minimum=0.0, maximum=1.0, value=0.2)
            origin_gif.append(origin)
            destination_gif.append(destination)
            thresholds_gif.append(threshold)
            face_tabs_gif.append(tab)

        face_mode_gif.change(fn=lambda mode: toggle_tabs_and_faces(mode, face_tabs_gif, origin_gif), inputs=[face_mode_gif], outputs=face_tabs_gif + origin_gif)
        demo.load(fn=lambda: toggle_tabs_and_faces("Single Face", face_tabs_gif, origin_gif), inputs=None, outputs=face_tabs_gif + origin_gif)

        gif_btn.click(fn=run, inputs=[gif_input] + origin_gif + destination_gif + thresholds_gif + [preview_checkbox_gif, face_mode_gif, partial_reface_ratio_gif, use_cache_gif], outputs=[gif_output, gif_file_output])

        gif_input.change(fn=lambda filepath: extract_faces_auto(filepath, refacer, max_faces=num_faces), inputs=gif_input, outputs=origin_gif)
        gif_input.change(fn=lambda file: file, inputs=gif_input, outputs=[gif_preview])
        gif_input.change(fn=lambda _: 0.0, inputs=gif_input, outputs=partial_reface_ratio_gif)

        
    # --- TIF MODE ---
    with gr.Tab("TIFF Mode"):
        with gr.Row():
            tif_input = gr.File(label="Original TIF", file_types=[".tif", ".tiff"])
            tif_preview = gr.Image(label="TIF Preview (Cover Page)", type="filepath")
            tif_output_preview = gr.Image(label="Refaced TIF Preview (Cover Page)", type="filepath")
            tif_output_file = gr.File(label="Refaced TIF (Download)", interactive=False)

        with gr.Row():
            face_mode_tif = gr.Radio(
                choices=["Single Face", "Multiple Faces", "Faces By Match"],
                value="Single Face",
                label="Replacement Mode"
            )
            partial_reface_ratio_tif = gr.Slider(label="Reface Ratio (0 = Full Face, 0.5 = Half Face)", minimum=0.0, maximum=0.5, value=0.0, step=0.1)
            tif_btn = gr.Button("Reface TIF", variant="primary")

        origin_tif, destination_tif, thresholds_tif, face_tabs_tif = [], [], [], []

        for i in range(num_faces):
            with gr.Tab(f"Face #{i+1}") as tab:
                with gr.Row():
                    origin = gr.Image(label="Face to replace")
                    destination = gr.Image(label="Destination face")
                threshold = gr.Slider(label="Threshold", minimum=0.0, maximum=1.0, value=0.2)
            origin_tif.append(origin)
            destination_tif.append(destination)
            thresholds_tif.append(threshold)
            face_tabs_tif.append(tab)

        face_mode_tif.change(
            fn=lambda mode: toggle_tabs_and_faces(mode, face_tabs_tif, origin_tif),
            inputs=[face_mode_tif],
            outputs=face_tabs_tif + origin_tif
        )

        demo.load(
            fn=lambda: toggle_tabs_and_faces("Single Face", face_tabs_tif, origin_tif),
            inputs=None,
            outputs=face_tabs_tif + origin_tif
        )

        def process_tif(tif_path, *vars):
            original_img = Image.open(tif_path)
            if hasattr(original_img, "n_frames") and original_img.n_frames > 1:
                original_img.seek(0)
            temp_preview_path = os.path.join("./tmp", f"tif_preview_{int(time.time() * 1000)}.jpg")
            original_img.convert('RGB').save(temp_preview_path)

            refaced_path = run_image(tif_path, *vars)

            refaced_img = Image.open(refaced_path)
            if hasattr(refaced_img, "n_frames") and refaced_img.n_frames > 1:
                refaced_img.seek(0)
            temp_refaced_preview_path = os.path.join("./tmp", f"refaced_tif_preview_{int(time.time() * 1000)}.jpg")
            refaced_img.convert('RGB').save(temp_refaced_preview_path)

            return temp_preview_path, temp_refaced_preview_path, refaced_path

        tif_btn.click(
            fn=lambda tif_path, *args: process_tif(tif_path, *args),
            inputs=[tif_input] + origin_tif + destination_tif + thresholds_tif + [face_mode_tif, partial_reface_ratio_tif],
            outputs=[tif_preview, tif_output_preview, tif_output_file]
        )

        tif_input.change(
            fn=lambda filepath: extract_faces_auto(filepath, refacer, max_faces=num_faces),
            inputs=tif_input,
            outputs=origin_tif
        )

        tif_input.change(
            fn=handle_tif_preview,
            inputs=tif_input,
            outputs=tif_preview
        )
        
        tif_input.change(fn=lambda _: 0.0, inputs=tif_input, outputs=partial_reface_ratio_tif)


    # --- VIDEO MODE ---
    with gr.Tab("Video Mode"):
        with gr.Row():
            video_input = gr.Video(label="Original video", format="mp4")
            video_output = gr.Video(label="Refaced Video", interactive=False, format="mp4")

        with gr.Row():
            face_mode_video = gr.Radio(
                choices=["Single Face", "Multiple Faces", "Faces By Match"],
                value="Single Face",
                label="Replacement Mode"
            )
            partial_reface_ratio_video = gr.Slider(label="Reface Ratio (0 = Full Face, 0.5 = Half Face)", minimum=0.0, maximum=0.5, value=0.0, step=0.1)
            video_btn = gr.Button("Reface Video", variant="primary")

        with gr.Row():
            rotate_left_btn = gr.Button("↺ Rotate Left", variant="secondary")
            rotate_right_btn = gr.Button("↻ Rotate Right", variant="secondary")
            redetect_btn = gr.Button("🔄 Re-detect Faces", variant="secondary", visible=False)

        with gr.Row():
            preview_checkbox_video = gr.Checkbox(label="Preview Generation (skip 90% of frames)", value=False)
            use_cache_video = gr.Checkbox(label="Enable Cache (Faster subsequent runs)", value=False)

        # Video History Section
        with gr.Accordion("📜 Video History", open=False):
            history_display = gr.HTML(label="Recent Video Swaps")
            refresh_history_btn = gr.Button("🔄 Refresh History", variant="secondary")
            clear_history_btn = gr.Button("🗑️ Clear History", variant="secondary")

        refresh_history_btn.click(
            fn=get_video_history,
            inputs=[],
            outputs=[history_display]
        )

        clear_history_btn.click(
            fn=clear_video_history,
            inputs=[],
            outputs=[history_display]
        )

        origin_video, destination_video, thresholds_video, face_tabs_video = [], [], [], []

        for i in range(num_faces):
            with gr.Tab(f"Face #{i+1}") as tab:
                with gr.Row():
                    origin = gr.Image(label="Face to replace")
                    destination = gr.Gallery(label="Destination face(s)", columns=4, height="auto", object_fit="contain", file_types=["image"])
                threshold = gr.Slider(label="Threshold", minimum=0.0, maximum=1.0, value=0.2)
            origin_video.append(origin)
            destination_video.append(destination)
            thresholds_video.append(threshold)
            face_tabs_video.append(tab)

        face_mode_video.change(
            fn=lambda mode: toggle_tabs_and_faces(mode, face_tabs_video, origin_video),
            inputs=[face_mode_video],
            outputs=face_tabs_video + origin_video
        )

        demo.load(
            fn=lambda: toggle_tabs_and_faces("Single Face", face_tabs_video, origin_video),
            inputs=None,
            outputs=face_tabs_video + origin_video
        )
        
        video_input.change(
            fn=lambda filepath: extract_faces_auto(filepath, refacer, max_faces=num_faces, isvideo=True),
            inputs=video_input,
            outputs=origin_video
        )
        
        video_input.change(fn=lambda _: 0.0, inputs=video_input, outputs=partial_reface_ratio_video)

        def handle_rotate_left(video_path):
            rotated = rotate_video(video_path, 'left')
            return gr.update(value=rotated)

        def handle_rotate_right(video_path):
            rotated = rotate_video(video_path, 'right')
            return gr.update(value=rotated)

        def handle_redetect(video_path):
            """Re-detect faces from the current (possibly rotated) video"""
            faces = extract_faces_auto(video_path, refacer, max_faces=num_faces, isvideo=True)
            return faces

        def update_redetect_visibility(mode):
            """Show Re-detect button only in Faces By Match mode"""
            return gr.update(visible=(mode == "Faces By Match"))

        face_mode_video.change(
            fn=update_redetect_visibility,
            inputs=face_mode_video,
            outputs=redetect_btn
        )

        rotate_left_btn.click(
            fn=handle_rotate_left,
            inputs=video_input,
            outputs=video_input
        )

        rotate_right_btn.click(
            fn=handle_rotate_right,
            inputs=video_input,
            outputs=video_input
        )

        redetect_btn.click(
            fn=handle_redetect,
            inputs=video_input,
            outputs=origin_video
        )

        def run_with_history_update(*args):
            for mp4_path, gif_path in run(*args):
                yield mp4_path, gif_path, get_video_history()

        video_btn.click(
            fn=run_with_history_update,
            inputs=[video_input] + origin_video + destination_video + thresholds_video + [preview_checkbox_video, face_mode_video, partial_reface_ratio_video, use_cache_video],
            outputs=[video_output, gr.File(visible=False), history_display]
        )

    # --- System / Cache Settings (Global) ---
    with gr.Accordion("⚙️ System Settings", open=False):
        with gr.Row():
            clear_cache_btn = gr.Button("🗑️ Clear Cache Directory", variant="secondary")
            clear_cache_status = gr.Textbox(label="Status", interactive=False, placeholder="Click button to clear cache")
            
        def trigger_clear_cache():
            try:
                refacer.clear_cache()
                return "Cache directory successfully cleared!"
            except Exception as e:
                return f"Error clearing cache: {e}"
                
        clear_cache_btn.click(fn=trigger_clear_cache, inputs=[], outputs=[clear_cache_status])

# --- ngrok connect (optional) ---
if args.ngrok:
    def connect(token, port, options):
        try:
            public_url = ngrok.connect(f"127.0.0.1:{port}", **options).url()
            print(f'ngrok URL: {public_url}')
        except Exception as e:
            print(f'ngrok connection aborted: {e}')

    connect(args.ngrok, args.server_port, {'region': args.ngrok_region, 'authtoken_from_env': False})

# --- Launch app ---
is_colab = False
try:
    __import__('google.colab')
    is_colab = True
except ImportError:
    pass

share_app = args.share_gradio or is_colab
if is_colab:
    print("[INFO] Google Colab environment detected. Automatically enabling Gradio sharing link.")

demo.queue().launch(favicon_path="icon.png", show_error=True, share=share_app, server_name=args.server_name, server_port=args.server_port)
