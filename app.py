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
import traceback
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

def _load_identity_profile_face(profile_file, slot_label=""):
    """Resolves a Face-slot's uploaded .npz into a synthetic Face, or None.

    A corrupted/incompatible .npz must not abort every other configured
    Face-slot: _build_video_face_jobs runs once, before run()'s per-job
    try/except, so an uncaught exception here would drop all jobs (including
    valid ones on other slots) instead of just this slot's.
    """
    path = _resolve_history_path(profile_file)
    if not path or not os.path.exists(path):
        return None
    try:
        profile = import_profile(path)
    except ValueError as e:
        gr.Warning(f"Perfil de identidade inválido em {slot_label or 'um slot'}: {e}. Ignorando este slot.")
        return None
    return profile["face"]

def _build_video_face_jobs(origins, destinations, thresholds, multiple_faces_mode, identity_profiles=None):
    """Build one job per destination face, keeping Multiple Faces as a single combined job.

    identity_profiles (optional): one entry per Face-slot; when a slot has an
    uploaded .npz identity profile, it takes priority over that slot's
    Destination Gallery (the gallery is also disabled in the UI when a
    profile is uploaded, see Video Mode tab).
    """
    identity_profiles = identity_profiles or [None] * num_faces

    if multiple_faces_mode:
        # All destination faces are swapped together, by position, in one video.
        faces = []
        labels = []
        for k in range(num_faces):
            profile_face = _load_identity_profile_face(identity_profiles[k], slot_label=f'Face #{k + 1}')
            if profile_face is not None:
                faces.append({'origin': None, 'identity_profile': profile_face, 'threshold': 0.0})
                labels.append(f'Face #{k + 1} (identity profile)')
                continue

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
        profile_face = _load_identity_profile_face(identity_profiles[k], slot_label=f'Face #{k + 1}')
        dest_files = destinations[k]
        has_destination = dest_files is not None and not (isinstance(dest_files, list) and len(dest_files) == 0)

        if profile_face is None and not has_destination:
            # Nothing configured for this slot — skip decoding origin_image,
            # matching the pre-identity-profile behavior of not doing any
            # work for an empty slot.
            continue

        origin_image = load_face_image(origins[k])

        if profile_face is not None:
            jobs.append({
                'faces': [{
                    'origin': origin_image,
                    'identity_profile': profile_face,
                    'threshold': thresholds[k]
                }],
                'label': f'Face #{k + 1} (identity profile)'
            })
            continue

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

def run(progress=gr.Progress(track_tqdm=True), *vars):
    video_path = vars[0]
    origins = vars[1:(num_faces+1)]
    destinations = vars[(num_faces+1):(num_faces*2)+1]
    thresholds = vars[(num_faces*2)+1:(num_faces*3)+1]
    identity_profiles = vars[(num_faces*3)+1:-5]
    preview = vars[-5]
    face_mode = vars[-4]
    partial_reface_ratio = vars[-3]
    oval_mask = vars[-2]
    use_cache = vars[-1]
    partial_blend_shape = "oval" if oval_mask else "rect"

    disable_similarity = (face_mode in ["Single Face", "Multiple Faces"])
    multiple_faces_mode = (face_mode == "Multiple Faces")

    jobs = _build_video_face_jobs(origins, destinations, thresholds, multiple_faces_mode, identity_profiles)

    if not jobs:
        yield None, None
        return

    last_mp4_path, last_gif_path = None, None

    # When the same target video will be processed for more than one face job,
    # decode + detect it once into a local variable and hand that to every
    # reface() call, instead of relying on refacer's hash-keyed RAM caches.
    # Not gated on use_cache: this is a run-local variable, not a persistent
    # cache — without it every extra job pays a full decode+detect+embedding
    # pass over the same video. The RAM ceiling inside
    # analyze_video_in_memory already falls back to per-job processing when
    # the video is too large to hold decoded in memory.
    precomputed = None
    if len(jobs) > 1:
        precomputed = refacer.analyze_video_in_memory(video_path, preview=preview)

    failed_jobs = []
    for k, job in enumerate(jobs):
        progress(k / len(jobs), desc=f"{job['label']} ({k + 1}/{len(jobs)})")
        try:
            mp4_path, gif_path = refacer.reface(
                video_path,
                job['faces'],
                preview=preview,
                disable_similarity=disable_similarity,
                multiple_faces_mode=multiple_faces_mode,
                partial_reface_ratio=partial_reface_ratio,
                partial_blend_shape=partial_blend_shape,
                use_cache=use_cache,
                precomputed=precomputed
            )
        except Exception as e:
            # A bad destination image (or any per-job failure) must not
            # silently abort the remaining queued jobs.
            print(f"[ERROR] Job '{job['label']}' failed: {e}")
            traceback.print_exc()
            failed_jobs.append(job['label'])
            continue

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

    if failed_jobs:
        gr.Warning(f"Falha ao processar: {', '.join(failed_jobs)}. Os demais jobs foram concluídos.")

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

# --- Identity Profile (multi-image / multi-video, single person) ---
import identity_profile
from identity_profile import IdentityProfileBuilder, export_profile, import_profile, merge_profiles

_IDENTITY_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
_IDENTITY_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".gif"}

def _pluralize_count(count, singular_suffix="", plural_suffix="s"):
    """`f"{count} amostra{_pluralize_count(count)}"` style helper — shared by
    extract/merge/discard status messages instead of each one repeating its
    own "(s)"/"(is)" logic.
    """
    return singular_suffix if count == 1 else plural_suffix

def _populate_builder_from_files(builder, source_files):
    """Feeds each uploaded file into the builder as an image or video sample
    source, based on its extension.
    """
    for f in source_files:
        path = _resolve_history_path(f)
        if not path or not os.path.exists(path):
            continue

        label = os.path.basename(path)
        ext = os.path.splitext(path)[1].lower()
        if ext in _IDENTITY_VIDEO_EXTENSIONS:
            builder.add_video(path, label)
        else:
            frame = cv2.imread(path)
            builder.add_image(frame, label)

def _format_extraction_status(profiles):
    status_lines = [
        f"{len(profiles)} pessoa(s) detectada(s), separada(s) automaticamente por similaridade facial.",
        "Se a mesma pessoa foi dividida em dois perfis, use \"Mesclar Perfis\" abaixo. "
        "Se um perfil é ruído/pessoa indesejada, use \"Descartar Perfil\". Se duas pessoas "
        "diferentes foram fundidas no mesmo perfil, extraia novamente com material separado "
        "por pessoa (mesclagem não separa um perfil já fundido).",
    ]
    for profile in profiles:
        n = profile['n_samples']
        status_lines.append(f"- {profile['name']}: {n} amostra{_pluralize_count(n)} válida{_pluralize_count(n)}")
    if profiles[0]["discarded"]:
        status_lines.append(f"Descartes ({profiles[0]['n_discarded']}):")
        status_lines.extend(f"  - {d['source']}: {d['reason']}" for d in profiles[0]["discarded"])
    return "\n".join(status_lines)

def extract_identity_profile(source_files):
    """Builds one reusable identity profile (Face + centroid embedding) per
    person detected in the uploaded images/videos (greedy clustering by
    cosine similarity — see identity_profile.cluster_samples). Video mode
    only consumes a profile via a per-slot .npz upload (see Video Mode tab)
    — no in-session sharing/gr.State is used across tabs, since Colab
    sessions are volatile and export is the primary persistence mechanism
    (see PLANO_IDENTIDADE_MULTI_FONTE.md).
    """
    if not source_files:
        empty_dropdown = gr.update(choices=[], value=None)
        return "Nenhum arquivo enviado.", [], empty_dropdown, empty_dropdown, empty_dropdown, []

    builder = IdentityProfileBuilder.from_refacer(refacer)
    _populate_builder_from_files(builder, source_files)

    if not builder.samples:
        reasons = "; ".join(f"{d['source']}: {d['reason']}" for d in builder.discarded) or "motivo desconhecido"
        raise gr.Error(f"Nenhuma amostra de rosto válida foi extraída. Descartes: {reasons}")

    profiles = builder.build_profiles()
    status = _format_extraction_status(profiles)
    choices = [profile["name"] for profile in profiles]

    # A lista de perfis (cada um com seu Face sintético) fica só na gr.State
    # desta aba, para o dropdown escolher qual baixar/testar sem precisar
    # re-extrair do zero. Não é persistida em disco além do(s) .npz baixado(s).
    return (
        status,
        _profiles_to_gallery(profiles),
        gr.update(choices=choices, value=choices[0]),
        gr.update(choices=choices, value=None),
        gr.update(choices=choices, value=None),
        profiles,
    )

def _get_selected_profile(profiles, selected_name):
    if not profiles or not selected_name:
        return None
    for profile in profiles:
        if profile["name"] == selected_name:
            return profile
    return None

def _profiles_to_gallery(profiles):
    gallery_items = []
    for profile in profiles:
        thumbnail = profile["thumbnail"]
        thumbnail_rgb = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB) if thumbnail is not None else None
        n = profile['n_samples']
        caption = f"{profile['name']} ({n} amostra{_pluralize_count(n)})"
        gallery_items.append((thumbnail_rgb, caption))
    return gallery_items

def merge_selected_profiles(profiles, name_a, name_b):
    """Combines two clustered profiles into one — the mitigation for the
    clustering greedily splitting the same person into two profiles (e.g.
    with/without glasses, very different pose/lighting between samples).
    Not intended to fix the opposite failure (two different people merged
    into one profile by the automatic clustering) — that requires discarding
    and re-extracting with cleaner/separated source material instead.
    """
    if not profiles:
        raise gr.Error("Nenhum perfil para mesclar. Extraia uma identidade primeiro.")
    if not name_a or not name_b:
        raise gr.Error("Selecione os dois perfis a mesclar.")
    if name_a == name_b:
        raise gr.Error("Selecione dois perfis diferentes para mesclar.")

    profile_a = _get_selected_profile(profiles, name_a)
    profile_b = _get_selected_profile(profiles, name_b)
    if profile_a is None or profile_b is None:
        raise gr.Error("Perfil selecionado não encontrado.")

    try:
        merged = merge_profiles(profile_a, profile_b, name=profile_a["name"])
    except ValueError as e:
        raise gr.Error(str(e))
    remaining = [p for p in profiles if p["name"] not in (name_a, name_b)]
    new_profiles = remaining + [merged]

    n = merged['n_samples']
    status = (
        f"{name_a} e {name_b} mesclados em {merged['name']} "
        f"({n} amostra{_pluralize_count(n)} combinada{_pluralize_count(n)})."
    )
    choices = [p["name"] for p in new_profiles]
    return (
        status,
        _profiles_to_gallery(new_profiles),
        gr.update(choices=choices, value=merged["name"]),
        gr.update(choices=choices, value=None),
        new_profiles,
    )

def discard_selected_profile(profiles, selected_name):
    """Removes a profile the clustering created spuriously (background
    face, misdetection, duplicate split) without needing to re-extract
    everything from scratch.
    """
    if not profiles or not selected_name:
        raise gr.Error("Nenhum perfil selecionado para descartar.")

    new_profiles = [p for p in profiles if p["name"] != selected_name]
    if len(new_profiles) == len(profiles):
        raise gr.Error("Perfil selecionado não encontrado.")

    n = len(new_profiles)
    profile_word = "perfil" if n == 1 else "perfis"
    status = f"{selected_name} descartado. {n} {profile_word} restante{_pluralize_count(n)}."
    choices = [p["name"] for p in new_profiles]
    next_selected = choices[0] if choices else None
    return (
        status,
        _profiles_to_gallery(new_profiles),
        gr.update(choices=choices, value=next_selected),
        gr.update(choices=choices, value=None),
        new_profiles,
    )

def export_selected_profile(profiles, selected_name):
    profile = _get_selected_profile(profiles, selected_name)
    if profile is None:
        raise gr.Error("Nenhum perfil selecionado. Extraia uma identidade primeiro.")

    export_path = os.path.join(
        "./tmp",
        f"identity_profile_{selected_name.replace(' ', '_')}_{int(time.time() * 1000)}.npz",
    )
    export_profile(profile, export_path)
    return gr.update(value=export_path, visible=True)

def preview_identity_swap(test_image_file, profiles, selected_name):
    """Applies the selected in-session identity profile to a separate test
    image, so the user can see the real swap result (profile identity ->
    someone else's face) before downloading/trusting the .npz, instead of
    judging it against the profile's own source photos (which would always
    look trivially perfect and prove nothing about swap quality).
    """
    profile = _get_selected_profile(profiles, selected_name)
    if profile is None:
        raise gr.Error("Extraia um perfil de identidade primeiro.")

    path = _resolve_history_path(test_image_file)
    if not path or not os.path.exists(path):
        raise gr.Error("Envie uma imagem de teste (o rosto que será substituído pelo perfil).")

    frame = cv2.imread(path)
    if frame is None:
        raise gr.Error("Não foi possível ler a imagem de teste.")

    # refacer is a single global instance also used by the video/image reface
    # pipeline (run()/run_image()) — hold swap_lock for the whole
    # prepare_faces+process_faces span so a concurrent video reface from
    # another tab can't mutate self.replacement_faces mid-preview.
    with refacer.swap_lock:
        refacer.prepare_faces(
            [{"identity_profile": profile["face"], "threshold": 0.0}],
            disable_similarity=True,
            multiple_faces_mode=False,
        )
        # partial_reface_ratio/partial_blend_shape are instance state left
        # over from any earlier video reface in this session (oval mask,
        # reface ratio) — reset them so the preview shows a plain full-face
        # swap, matching what the user is actually judging the profile against.
        refacer.partial_reface_ratio = 0.0
        refacer.partial_blend_shape = "rect"
        swapped = refacer.process_faces(frame.copy())
    return cv2.cvtColor(swapped, cv2.COLOR_BGR2RGB)

def _reset_profile_dropdown(profiles):
    """Resets a profile-selection dropdown's choices after the profile list
    changes (merge/discard), clearing any stale selection.
    """
    return gr.update(choices=[p["name"] for p in profiles], value=None)

def clear_identity_temp_files():
    for fname in os.listdir("./tmp"):
        if fname.startswith("identity_profile_"):
            try:
                os.remove(os.path.join("./tmp", fname))
            except Exception as e:
                print(f"Warning: could not delete {fname}: {e}")
    return "Arquivos temporários de identidade removidos."

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


    # --- CRIAR IDENTIDADE ---
    with gr.Tab("Criar Identidade"):
        gr.Markdown(
            "**Uso somente com autorização.** Embeddings faciais são dados biométricos "
            "sensíveis. Processamento é feito localmente nesta sessão — nenhuma imagem, "
            "vídeo ou embedding é enviado para serviços externos. As pessoas presentes no "
            "material são separadas automaticamente por similaridade facial (\"Pessoa 1\", "
            "\"Pessoa 2\", ...) — **a separação automática pode errar** (fundir duas pessoas "
            "parecidas, ou dividir uma pessoa em dois perfis); confira a galeria de "
            "resultados abaixo antes de confiar em um perfil. Use \"Mesclar Perfis\" se a "
            "mesma pessoa foi dividida em dois, ou \"Descartar Perfil\" para remover ruído "
            "— fusão indevida de pessoas diferentes ainda exige reextrair com material "
            "separado. De vídeos, uma amostra de quadros é "
            f"extraída automaticamente (a cada {identity_profile.VIDEO_FRAME_STRIDE} frames, "
            f"até {identity_profile.MAX_FRAMES_PER_VIDEO} por vídeo)."
        )
        identity_consent = gr.Checkbox(
            label="Confirmo que possuo autorização para processar este material.",
            value=False,
        )
        identity_images = gr.File(
            label="Imagens e/ou vídeos (múltiplos arquivos, podem conter várias pessoas)",
            file_count="multiple",
            file_types=sorted(_IDENTITY_IMAGE_EXTENSIONS | _IDENTITY_VIDEO_EXTENSIONS),
            interactive=False,
        )
        identity_extract_btn = gr.Button("Extrair Identidade(s)", variant="primary", interactive=False)

        identity_status = gr.Textbox(label="Status da extração", lines=8, interactive=False)
        identity_gallery = gr.Gallery(
            label="Pessoas detectadas (amostra representativa de cada perfil)",
            columns=4,
            height="auto",
            object_fit="contain",
        )

        identity_profile_state = gr.State([])
        identity_selected_profile = gr.Dropdown(label="Perfil ativo (para baixar/testar)", choices=[], value=None)

        with gr.Accordion("Corrigir agrupamento (mesclar/descartar perfis)", open=False):
            gr.Markdown(
                "Se a separação automática dividiu a **mesma pessoa** em dois perfis "
                "(ex.: com/sem óculos, ângulos muito diferentes), mescle-os abaixo — o "
                "perfil combinado recalcula o centroide usando as amostras de ambos. "
                "Mesclar **não separa** dois perfis já mesclados nem corrige duas pessoas "
                "diferentes que foram fundidas em um só — nesse caso, descarte e reextraia "
                "com o material separado por pessoa."
            )
            with gr.Row():
                identity_merge_a = gr.Dropdown(label="Perfil A", choices=[], value=None)
                identity_merge_b = gr.Dropdown(label="Perfil B", choices=[], value=None)
            identity_merge_btn = gr.Button("Mesclar Perfis", variant="secondary")
            identity_discard_btn = gr.Button("🗑️ Descartar Perfil Selecionado", variant="secondary")

        gr.Markdown(
            "⚠️ O arquivo exportado contém dados biométricos derivados de rosto(s). "
            "Trate como informação sensível — não compartilhe sem autorização. "
            "Baixe-o agora: o Colab não retém a sessão entre execuções."
        )
        identity_export_btn = gr.Button("Baixar Perfil Selecionado (.npz)", variant="primary")
        identity_export_file = gr.File(label="Perfil de identidade (.npz)", visible=False, interactive=False)

        gr.Markdown(
            "**Teste antes de confiar no perfil**: envie uma foto de teste com o rosto "
            "de **outra pessoa** (não a que você acabou de extrair) para ver o resultado "
            "real do swap com o perfil selecionado acima — testar na própria foto de "
            "origem sempre pareceria perfeito e não comprova nada."
        )
        with gr.Row():
            identity_test_image = gr.Image(label="Imagem de teste (outro rosto)", type="filepath")
            identity_test_output = gr.Image(label="Resultado do swap com o perfil", interactive=False)
        identity_test_btn = gr.Button("Testar Perfil Selecionado", variant="secondary")

        identity_cleanup_btn = gr.Button("🗑️ Apagar arquivos temporários de identidade", variant="secondary")
        identity_cleanup_status = gr.Textbox(label="", interactive=False, show_label=False)

        identity_consent.change(
            fn=lambda consent: (gr.update(interactive=consent), gr.update(interactive=consent)),
            inputs=[identity_consent],
            outputs=[identity_images, identity_extract_btn],
        )

        identity_extract_btn.click(
            fn=extract_identity_profile,
            inputs=[identity_images],
            outputs=[
                identity_status,
                identity_gallery,
                identity_selected_profile,
                identity_merge_a,
                identity_merge_b,
                identity_profile_state,
            ],
        )

        identity_export_btn.click(
            fn=export_selected_profile,
            inputs=[identity_profile_state, identity_selected_profile],
            outputs=[identity_export_file],
        )

        identity_test_btn.click(
            fn=preview_identity_swap,
            inputs=[identity_test_image, identity_profile_state, identity_selected_profile],
            outputs=[identity_test_output],
        )

        identity_merge_btn.click(
            fn=merge_selected_profiles,
            inputs=[identity_profile_state, identity_merge_a, identity_merge_b],
            outputs=[
                identity_status,
                identity_gallery,
                identity_selected_profile,
                identity_merge_a,
                identity_profile_state,
            ],
        ).then(
            fn=_reset_profile_dropdown,
            inputs=[identity_profile_state],
            outputs=[identity_merge_b],
        )

        identity_discard_btn.click(
            fn=discard_selected_profile,
            inputs=[identity_profile_state, identity_selected_profile],
            outputs=[
                identity_status,
                identity_gallery,
                identity_selected_profile,
                identity_merge_a,
                identity_profile_state,
            ],
        ).then(
            fn=_reset_profile_dropdown,
            inputs=[identity_profile_state],
            outputs=[identity_merge_b],
        )

        identity_cleanup_btn.click(
            fn=clear_identity_temp_files,
            inputs=[],
            outputs=[identity_cleanup_status],
        )

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
            oval_mask_video = gr.Checkbox(label="Oval Mask (lip-to-chin, preserves cheeks)", value=False)
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
        identity_profile_video = []

        for i in range(num_faces):
            with gr.Tab(f"Face #{i+1}") as tab:
                with gr.Row():
                    origin = gr.Image(label="Face to replace")
                    destination = gr.Gallery(label="Destination face(s)", columns=4, height="auto", object_fit="contain", file_types=["image"])
                threshold = gr.Slider(label="Threshold", minimum=0.0, maximum=1.0, value=0.2)
                identity_profile_file = gr.File(
                    label="Identity Profile (.npz) — opcional, substitui a galeria acima quando enviado",
                    file_count="single",
                    file_types=[".npz"],
                )
            origin_video.append(origin)
            destination_video.append(destination)
            thresholds_video.append(threshold)
            face_tabs_video.append(tab)
            identity_profile_video.append(identity_profile_file)

            identity_profile_file.change(
                fn=lambda f: gr.update(interactive=(f is None)),
                inputs=[identity_profile_file],
                outputs=[destination],
            )

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

        def run_with_history_update(progress=gr.Progress(track_tqdm=True), *args):
            # Gradio injects Progress into the registered handler (this one),
            # not into plain Python calls — forward it to run() explicitly so
            # the video tab's inputs don't shift into the progress slot.
            for mp4_path, gif_path in run(progress, *args):
                yield mp4_path, gif_path, get_video_history()

        video_btn.click(
            fn=run_with_history_update,
            inputs=[video_input] + origin_video + destination_video + thresholds_video + identity_profile_video + [preview_checkbox_video, face_mode_video, partial_reface_ratio_video, oval_mask_video, use_cache_video],
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
