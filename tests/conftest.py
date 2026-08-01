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
    # identity_profile.py imports cv2 too but does real image-processing math
    # on plain numpy arrays (Laplacian variance, Gaussian blur, resize) that
    # doesn't need a GPU/native backend — a real, lightweight implementation
    # here (numpy-only, no actual cv2 dependency) lets those functions be
    # unit-tested for real instead of only asserting they don't crash.
    def _gaussian_blur(src, ksize, sigma_x, sigma_y=0):
        import numpy as np
        sigma_x = sigma_x if sigma_x > 0 else 1.0
        sigma_y = sigma_y if sigma_y > 0 else sigma_x
        radius_x = max(1, int(3 * sigma_x))
        radius_y = max(1, int(3 * sigma_y))
        xs = np.arange(-radius_x, radius_x + 1)
        ys = np.arange(-radius_y, radius_y + 1)
        kx = np.exp(-(xs ** 2) / (2 * sigma_x ** 2))
        ky = np.exp(-(ys ** 2) / (2 * sigma_y ** 2))
        kx /= kx.sum()
        ky /= ky.sum()

        def _convolve_1d(arr, kernel, axis):
            padded = np.pad(arr, [(len(kernel) // 2, len(kernel) // 2) if i == axis else (0, 0)
                                   for i in range(arr.ndim)], mode="edge")
            out = np.zeros_like(arr)
            for i, w in enumerate(kernel):
                sl = [slice(None)] * arr.ndim
                sl[axis] = slice(i, i + arr.shape[axis])
                out += w * padded[tuple(sl)]
            return out

        result = _convolve_1d(src, ky, axis=0)
        result = _convolve_1d(result, kx, axis=1)
        return result

    def _laplacian(src, ddepth):
        import numpy as np
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        padded = np.pad(src.astype(np.float64), 1, mode="edge")
        out = np.zeros_like(src, dtype=np.float64)
        h, w = src.shape[:2]
        for dy in range(3):
            for dx in range(3):
                if kernel[dy, dx] == 0:
                    continue
                out += kernel[dy, dx] * padded[dy:dy + h, dx:dx + w]
        return out

    def _resize(src, dsize, interpolation=None):
        import numpy as np
        w, h = dsize
        src_h, src_w = src.shape[:2]
        if src_h == 0 or src_w == 0 or h == 0 or w == 0:
            return np.zeros((h, w) + src.shape[2:], dtype=src.dtype)
        row_idx = np.clip((np.arange(h) * src_h / h).astype(int), 0, src_h - 1)
        col_idx = np.clip((np.arange(w) * src_w / w).astype(int), 0, src_w - 1)
        return src[row_idx][:, col_idx]

    _COLOR_BGR2YCRCB = 36  # matches real cv2.COLOR_BGR2YCrCb value

    def _cvt_color(src, code):
        import numpy as np
        if code == _COLOR_BGR2YCRCB:
            # Same coefficients as real OpenCV's BGR<->YCrCb (ITU-R BT.601,
            # full range): Y = 0.299R + 0.587G + 0.114B; Cr/Cb centered at 128.
            b, g, r = src[..., 0].astype(np.float64), src[..., 1].astype(np.float64), src[..., 2].astype(np.float64)
            y = 0.299 * r + 0.587 * g + 0.114 * b
            cr = (r - y) * 0.713 + 128.0
            cb = (b - y) * 0.564 + 128.0
            return np.stack([y, cr, cb], axis=-1).astype(np.uint8 if src.dtype == np.uint8 else np.float64)
        if src.ndim == 2:
            return src
        # BGR2GRAY (any other code falls back to this — only path used besides YCrCb above).
        weights = np.array([0.114, 0.587, 0.299])  # B, G, R
        return (src.astype(np.float64) * weights).sum(axis=-1)

    def _bilateral_filter(src, d, sigma_color, sigma_space):
        # Reference-quality (not performance-optimized) implementation —
        # same math as OpenCV's bilateral filter: each output pixel is a
        # weighted average of its neighborhood, weighted by BOTH spatial
        # distance (Gaussian, sigma_space) and color/intensity difference
        # (Gaussian, sigma_color). The color term is what makes this filter
        # edge-preserving (a neighbor very different in value contributes
        # almost nothing, even if spatially close), so — unlike
        # GaussianBlur/DoG — it produces no ringing around isolated
        # high-contrast blobs. Two details matter for fidelity (verified
        # against real OpenCV 5.0 output, max abs diff 3e-4 with them vs
        # 2.3 gray levels without): OpenCV uses a CIRCULAR neighborhood
        # (neighbors with r > d//2 are skipped, not just down-weighted) and
        # BORDER_DEFAULT = reflect-101 (np.pad mode="reflect").
        import numpy as np
        src = src.astype(np.float64)
        h, w = src.shape
        radius = d // 2
        yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        spatial_weight = np.exp(-(xx.astype(np.float64) ** 2 + yy.astype(np.float64) ** 2) / (2 * sigma_space ** 2))
        spatial_weight *= (xx ** 2 + yy ** 2) <= radius ** 2
        padded = np.pad(src, radius, mode="reflect")

        result = np.zeros_like(src)
        for i in range(h):
            for j in range(w):
                patch = padded[i:i + d, j:j + d]
                color_weight = np.exp(-((patch - src[i, j]) ** 2) / (2 * sigma_color ** 2))
                weight = spatial_weight * color_weight
                result[i, j] = (patch * weight).sum() / weight.sum()
        return result.astype(np.float32)

    def _invert_affine_transform(m):
        import numpy as np
        m = np.asarray(m, dtype=np.float64)
        linear = m[:, :2]
        translation = m[:, 2]
        linear_inv = np.linalg.inv(linear)
        return np.concatenate([linear_inv, (-linear_inv @ translation)[:, np.newaxis]], axis=1).astype(np.float32)

    def _warp_affine(src, m, dsize, borderValue=0.0):
        # Inverse-mapping nearest-neighbor warp — good enough for tests that
        # only check "did the aligned/warped-back region land roughly where
        # geometry says it should", not pixel-perfect interpolation parity
        # with the real cv2.warpAffine (bilinear/bicubic).
        import numpy as np
        m = np.asarray(m, dtype=np.float64)
        w, h = dsize
        m_inv = _invert_affine_transform(m)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        src_x = m_inv[0, 0] * xx + m_inv[0, 1] * yy + m_inv[0, 2]
        src_y = m_inv[1, 0] * xx + m_inv[1, 1] * yy + m_inv[1, 2]
        src_h, src_w = src.shape[:2]
        valid = (src_x >= 0) & (src_x < src_w) & (src_y >= 0) & (src_y < src_h)
        src_xi = np.clip(src_x.astype(int), 0, src_w - 1)
        src_yi = np.clip(src_y.astype(int), 0, src_h - 1)

        out_shape = (h, w) + src.shape[2:]
        out = np.full(out_shape, borderValue, dtype=src.dtype)
        out[valid] = src[src_yi[valid], src_xi[valid]]
        return out

    def _convex_hull(points):
        # Monotone chain — standard O(n log n) convex hull, good enough for
        # the small point sets (mouth contour, ~13 points) exercised in tests.
        import numpy as np
        pts = sorted({(int(p[0]), int(p[1])) for p in points})
        if len(pts) <= 2:
            return np.array(pts, dtype=np.int32).reshape(-1, 1, 2)

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        lower = []
        for p in pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        hull = lower[:-1] + upper[:-1]
        return np.array(hull, dtype=np.int32).reshape(-1, 1, 2)

    def _fill_convex_poly(img, points, color):
        # Point-in-polygon via matplotlib-free even-odd rule — fine for test
        # fixtures, not meant to match cv2's exact rasterization pixel-for-pixel.
        import numpy as np
        poly = np.asarray(points).reshape(-1, 2)
        h, w = img.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        inside = np.zeros((h, w), dtype=bool)
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            crosses = ((y1 > yy) != (y2 > yy))
            denom = (y2 - y1) if (y2 - y1) != 0 else 1e-9
            x_intersect = x1 + (yy - y1) * (x2 - x1) / denom
            inside ^= crosses & (xx < x_intersect)
        img[inside] = color
        return img

    _ROTATE_90_CLOCKWISE = 0
    _ROTATE_180 = 1
    _ROTATE_90_COUNTERCLOCKWISE = 2

    def _rotate(src, rotate_code):
        import numpy as np
        if rotate_code == _ROTATE_90_CLOCKWISE:
            return np.rot90(src, k=-1).copy()
        if rotate_code == _ROTATE_180:
            return np.rot90(src, k=2).copy()
        if rotate_code == _ROTATE_90_COUNTERCLOCKWISE:
            return np.rot90(src, k=1).copy()
        raise ValueError(f"unsupported rotate_code: {rotate_code}")

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
        COLOR_BGR2GRAY=6,
        COLOR_BGR2YCrCb=_COLOR_BGR2YCRCB,
        CV_64F=6,
        INTER_LINEAR=1,
        ROTATE_90_CLOCKWISE=_ROTATE_90_CLOCKWISE,
        ROTATE_180=_ROTATE_180,
        ROTATE_90_COUNTERCLOCKWISE=_ROTATE_90_COUNTERCLOCKWISE,
        GaussianBlur=_gaussian_blur,
        Laplacian=_laplacian,
        resize=_resize,
        cvtColor=_cvt_color,
        warpAffine=_warp_affine,
        invertAffineTransform=_invert_affine_transform,
        bilateralFilter=_bilateral_filter,
        convexHull=_convex_hull,
        fillConvexPoly=_fill_convex_poly,
        rotate=_rotate,
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

    _make_submodule("insightface", "model_zoo", inswapper=None, landmark=None)
    _make_submodule("insightface.model_zoo", "inswapper", INSwapper=object)
    _make_submodule("insightface.model_zoo", "landmark", Landmark=object)

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

    # identity_profile.py (_extract_skin_texture) and refacer.py
    # (_apply_skin_texture) both need a REAL similarity-transform alignment
    # (not the real face_align.py, which needs the heavy skimage dependency
    # this test suite deliberately avoids installing) — a numpy-only
    # least-squares similarity fit (rotation+uniform scale+translation),
    # equivalent to skimage.transform.SimilarityTransform.estimate for the
    # 5-point case, covers the one template ('arcface', image_size=112) this
    # codebase actually calls estimate_norm/norm_crop with.
    import numpy as np

    arcface_src = np.array(
        [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
         [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32,
    )

    def _similarity_transform(src_pts, dst_pts):
        src_mean = src_pts.mean(axis=0)
        dst_mean = dst_pts.mean(axis=0)
        src_c = src_pts - src_mean
        dst_c = dst_pts - dst_mean
        cov = dst_c.T @ src_c / len(src_pts)
        u, s, vt = np.linalg.svd(cov)
        d = np.ones(2)
        if np.linalg.det(u) * np.linalg.det(vt) < 0:
            d[-1] = -1
        r = u @ np.diag(d) @ vt
        var_src = (src_c ** 2).sum() / len(src_pts)
        scale = (s * d).sum() / var_src if var_src > 0 else 1.0
        t = dst_mean - scale * r @ src_mean
        m = np.zeros((2, 3), dtype=np.float32)
        m[:, :2] = scale * r
        m[:, 2] = t
        return m

    def _estimate_norm(lmk, image_size=112, mode="arcface"):
        assert lmk.shape == (5, 2)
        src = arcface_src if image_size == 112 else float(image_size) / 112 * arcface_src
        return _similarity_transform(np.asarray(lmk, dtype=np.float32), src), 0

    def _norm_crop(img, landmark, image_size=112, mode="arcface"):
        import cv2
        m, _ = _estimate_norm(landmark, image_size, mode)
        return cv2.warpAffine(img, m, (image_size, image_size), borderValue=0.0)

    _install_stub(
        "face_align",
        estimate_norm=_estimate_norm,
        norm_crop=_norm_crop,
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
