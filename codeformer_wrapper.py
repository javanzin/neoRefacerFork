import os
import torch
import cv2
import numpy as np
from pathlib import Path
from torchvision.transforms.functional import normalize
from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from basicsr.utils.registry import ARCH_REGISTRY

# Lazy loading - models loaded only when needed
_device = None
_net = None
_face_helper = None
_models_loaded = False

def _load_models():
    """Lazy load CodeFormer models only when first needed."""
    global _device, _net, _face_helper, _models_loaded
    if _models_loaded:
        return  # Already loaded

    # Cross-platform device selection: CUDA > MPS > CPU
    if torch.cuda.is_available():
        _device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        _device = torch.device("mps")
    else:
        _device = torch.device("cpu")

    # Download and load model
    pretrain_model_url = {
        'restoration': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
    }

    _net = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                                           connect_list=['32', '64', '128', '256']).to(_device)

    ckpt_path = load_file_from_url(url=pretrain_model_url['restoration'],
                                    model_dir='weights/CodeFormer', progress=True, file_name=None)
    checkpoint = torch.load(ckpt_path, map_location=_device)['params_ema']
    _net.load_state_dict(checkpoint)
    _net.eval()

    _face_helper = FaceRestoreHelper(
        upscale_factor=1,
        face_size=512,
        crop_ratio=(1, 1),
        det_model='retinaface_resnet50',
        save_ext='jpg',
        use_parse=True,
        device=_device
    )

    _models_loaded = True
    print(f"[INFO] CodeFormer models loaded on device: {_device}")

def _enhance_img(img: np.ndarray, w: float = 0.5) -> np.ndarray:
    """
    Internal helper to enhance a numpy image with CodeFormer.
    """
    _load_models()  # Lazy load on first use

    _face_helper.clean_all()
    _face_helper.read_image(img)
    num_faces = _face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
    if num_faces == 0:
        return img  # Return original if no faces detected

    _face_helper.align_warp_face()

    for cropped_face in _face_helper.cropped_faces:
        cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True).to(_device)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t = cropped_face_t.unsqueeze(0)  # (1, 3, H, W), already on correct device

        with torch.no_grad():
            output = _net(cropped_face_t, w=w, adain=True)[0]
            restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))

        restored_face = restored_face.astype('uint8')
        _face_helper.add_restored_face(restored_face)

    _face_helper.get_inverse_affine(None)
    restored_img = _face_helper.paste_faces_to_input_image()
    return restored_img

def enhance_image(input_image_path: str, w: float = 0.5) -> str:
    """
    Enhances an input image using CodeFormer and saves it with a '.enhanced.jpg' suffix.
    """
    input_path = Path(input_image_path)
    output_path = input_path.with_name(f"{input_path.stem}.enhanced.jpg")

    img = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {input_image_path}")

    restored_img = _enhance_img(img, w=w)

    os.makedirs(output_path.parent, exist_ok=True)
    cv2.imwrite(str(output_path), restored_img)
    print(f"Enhanced image saved to: {output_path}")
    return str(output_path)

def enhance_image_memory(img: np.ndarray, w: float = 0.5) -> np.ndarray:
    """
    Enhances an input image entirely in memory and returns the enhanced image.
    """
    return _enhance_img(img, w=w)

def unload_models():
    """
    Unload CodeFormer models to free VRAM. Call this when done with image enhancement.
    """
    global _device, _net, _face_helper, _models_loaded
    if _models_loaded:
        del _net
        del _face_helper
        _net = None
        _face_helper = None
        _models_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[INFO] CodeFormer models unloaded from memory")