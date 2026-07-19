"""Stubs for the heavy, model/GPU-backed dependencies refacer.py imports at
module load time (cv2, onnxruntime, onnx, skimage, insightface, and the local
recognition/* wrappers, plus codeformer_wrapper).

Goal: let `import refacer` succeed on a plain machine with only numpy
installed, so the pure-logic pieces of refacer.py (blend math, batch sizing,
RAM budgeting, serial-vs-threaded dispatch) can be unit tested without
installing GPU/ML dependencies or Docker. This does NOT exercise real face
detection/swap — that still needs the real stack (Colab/Lightning) to verify.

If refacer.py grows a new top-level import from one of these heavy packages,
this file needs a matching stub or `import refacer` will fail here (while
still working fine in the real environment where the real package exists).
"""
import sys
import types


def _install_stub(name, **attrs):
    if name in sys.modules:
        return sys.modules[name]
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def _make_submodule(parent_name, child_name, **attrs):
    full_name = f"{parent_name}.{child_name}"
    child = _install_stub(full_name, **attrs)
    parent = sys.modules[parent_name]
    setattr(parent, child_name, child)
    return child


class _DummyCap:
    """Minimal stand-in for cv2.VideoCapture; only used so refacer.py
    methods can be called without opening a real video file. Tests that
    exercise video I/O paths should mock this further as needed."""

    def __init__(self, *a, **k):
        pass

    def isOpened(self):
        return False

    def read(self):
        return False, None

    def get(self, *_):
        return 0

    def set(self, *_):
        pass

    def release(self):
        pass


def _install_cv2_stub():
    cv2 = _install_stub(
        "cv2",
        CAP_FFMPEG=0,
        CAP_PROP_BUFFERSIZE=0,
        CAP_PROP_FRAME_COUNT=0,
        CAP_PROP_FPS=0,
        CAP_PROP_FRAME_WIDTH=0,
        CAP_PROP_FRAME_HEIGHT=0,
        VideoCapture=_DummyCap,
        VideoWriter=lambda *a, **k: None,
    )
    cv2.VideoWriter_fourcc = staticmethod(lambda *a, **k: 0)
    return cv2


def _install_onnxruntime_stub():
    class _SessionOptions:
        def __init__(self):
            self.execution_mode = None
            self.graph_optimization_level = None
            self.intra_op_num_threads = 1

    class _ExecutionMode:
        ORT_SEQUENTIAL = 0

    class _GraphOptimizationLevel:
        ORT_ENABLE_ALL = 0

    _install_stub(
        "onnxruntime",
        SessionOptions=_SessionOptions,
        ExecutionMode=_ExecutionMode,
        GraphOptimizationLevel=_GraphOptimizationLevel,
        InferenceSession=lambda *a, **k: None,
        get_available_providers=lambda: [],
        set_default_logger_severity=lambda *_: None,
        preload_dlls=lambda: None,
    )


def _install_insightface_stub():
    _install_stub("insightface")

    class _Face(dict):
        """insightface.app.common.Face is a dict-like attribute bag."""

        def __getattr__(self, item):
            try:
                return self[item]
            except KeyError as exc:
                raise AttributeError(item) from exc

        def __setattr__(self, key, value):
            self[key] = value

    _make_submodule("insightface", "app", common=None)
    _make_submodule("insightface.app", "common", Face=_Face)

    _make_submodule("insightface", "model_zoo", inswapper=None)
    _make_submodule("insightface.model_zoo", "inswapper", INSwapper=object)

    _make_submodule("insightface", "utils", storage=None)
    _make_submodule(
        "insightface.utils", "storage",
        ensure_available=lambda *a, **k: "",
    )


def _install_skimage_stub():
    _install_stub("skimage")
    _make_submodule("skimage", "transform", estimate_norm=lambda *a, **k: None)


def _install_local_recognition_stubs():
    # refacer.py does `sys.path.insert(1, './recognition')` then
    # `from scrfd import SCRFD` / `from arcface_onnx import ArcFaceONNX` /
    # `import face_align` — these are project-local modules (not third-party
    # packages), but they themselves import cv2/onnxruntime/onnx/skimage at
    # the top, so they must be stubbed too rather than imported for real.
    _install_stub("scrfd", SCRFD=object)
    _install_stub("arcface_onnx", ArcFaceONNX=object)
    _install_stub(
        "face_align",
        estimate_norm=lambda *a, **k: (None, None),
    )


def _install_codeformer_wrapper_stub():
    _install_stub(
        "codeformer_wrapper",
        enhance_image=lambda *a, **k: None,
        enhance_image_memory=lambda img, w=0.5: img,
    )


def _install_misc_stubs():
    _install_stub("onnx")
    _install_stub("ffmpeg", input=lambda *a, **k: None, probe=lambda *a, **k: {"streams": []})
    _install_stub("psutil", virtual_memory=lambda: types.SimpleNamespace(total=16 * 1024 ** 3))


def install_all_stubs():
    _install_cv2_stub()
    _install_onnxruntime_stub()
    _install_insightface_stub()
    _install_skimage_stub()
    _install_local_recognition_stubs()
    _install_codeformer_wrapper_stub()
    _install_misc_stubs()


install_all_stubs()
