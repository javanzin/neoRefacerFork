"""Testa a correção experimental de boca muito aberta/língua para fora
(_restore_original_mouth_if_open e auxiliares, ver refacer.OPEN_MOUTH_FIX_ENABLED
e MELHORIAS.md) — opt-in, reversível via feature flag.
"""
import numpy as np
import pytest

from insightface.app.common import Face

from refacer import Refacer


def _landmarks_106_with_mouth(top_y, bottom_y, left_x=40.0, right_x=72.0, center_x=56.0):
    """Array (106, 2) mínimo, só com os índices que _mouth_open_ratio e
    _mouth_polygon_mask realmente leem (52-70) preenchidos de forma
    plausível — os demais ficam em zero, irrelevantes para estes testes.

    Contorno externo (52-64) traçado como uma ELIPSE (não uma linha reta):
    fillConvexPoly/convexHull de pontos colineares produz área zero, o que
    não representa o contorno labial real (sempre tem alguma altura).
    """
    landmarks = np.zeros((106, 2), dtype=np.float32)
    outer_cx = (left_x + right_x) / 2.0
    outer_cy = (top_y + bottom_y) / 2.0
    outer_rx = (right_x - left_x) / 2.0
    outer_ry = max(3.0, (bottom_y - top_y) / 2.0 + 2.0)  # um pouco maior que a abertura interna
    n_outer = 13
    for i, idx in enumerate(range(52, 65)):
        angle = 2 * np.pi * i / n_outer
        landmarks[idx] = [outer_cx + outer_rx * np.cos(angle), outer_cy + outer_ry * np.sin(angle)]
    landmarks[52] = [left_x, outer_cy]
    landmarks[64] = [right_x, outer_cy]
    # Contorno interno (65-70): o que _mouth_open_ratio usa para abertura.
    landmarks[67] = [center_x, top_y]
    landmarks[70] = [center_x, bottom_y]
    return landmarks


def test_mouth_open_ratio_zero_with_closed_mouth():
    # Lábios internos encostados (top == bottom) -> abertura zero.
    landmarks = _landmarks_106_with_mouth(top_y=60.0, bottom_y=60.0)
    ratio = Refacer._mouth_open_ratio(Refacer, landmarks)
    assert ratio == pytest.approx(0.0, abs=1e-6)


def test_mouth_open_ratio_grows_with_gap():
    landmarks_small = _landmarks_106_with_mouth(top_y=58.0, bottom_y=60.0)
    landmarks_large = _landmarks_106_with_mouth(top_y=50.0, bottom_y=70.0)
    ratio_small = Refacer._mouth_open_ratio(Refacer, landmarks_small)
    ratio_large = Refacer._mouth_open_ratio(Refacer, landmarks_large)
    assert ratio_large > ratio_small


def test_mouth_open_ratio_handles_degenerate_zero_width_mouth():
    # Canto esquerdo == canto direito (detecção degenerada) não deve levantar
    # ZeroDivisionError nem retornar inf/nan.
    landmarks = _landmarks_106_with_mouth(top_y=55.0, bottom_y=65.0, left_x=50.0, right_x=50.0)
    ratio = Refacer._mouth_open_ratio(Refacer, landmarks)
    assert ratio == 0.0


def test_mouth_polygon_mask_is_zero_outside_polygon_and_positive_inside():
    landmarks = _landmarks_106_with_mouth(top_y=55.0, bottom_y=65.0)
    mask = Refacer._mouth_polygon_mask(Refacer, (112, 112, 3), landmarks, feather_px=5)
    assert mask.shape == (112, 112, 1)
    # Centro da boca (dentro do polígono) tem valor alto.
    assert mask[60, 56, 0] > 0.5
    # Canto oposto do crop (bem longe da boca) fica em zero.
    assert mask[5, 5, 0] == pytest.approx(0.0, abs=1e-6)


def test_mouth_polygon_mask_feathers_the_edge():
    # Não pode ser uma máscara binária — deve haver rampa suave na borda,
    # senão o corte lê como um contorno visível (mesmo motivo documentado em
    # identity_profile._skin_region_mask).
    landmarks = _landmarks_106_with_mouth(top_y=55.0, bottom_y=65.0)
    mask = Refacer._mouth_polygon_mask(Refacer, (112, 112, 3), landmarks, feather_px=9)
    fractional = ((mask > 0.0) & (mask < 1.0)).mean()
    assert fractional > 0.0


class _FakeLandmarkModel:
    def __init__(self, landmarks):
        self._landmarks = landmarks
        self.calls = 0

    def get(self, frame, face):
        self.calls += 1
        return self._landmarks


def _refacer_with_landmark_model(landmark_model):
    refacer = Refacer.__new__(Refacer)
    refacer.landmark_106 = landmark_model
    return refacer


def test_restore_original_mouth_noop_when_landmark_model_unavailable():
    # OPEN_MOUTH_FIX_ENABLED pode estar ligado mas o modelo pode não ter
    # carregado (ver __init_apps, fallback silencioso) — não deve quebrar,
    # deve devolver o frame trocado intocado.
    refacer = _refacer_with_landmark_model(None)
    face = Face()
    face.bbox = np.array([0.0, 0.0, 112.0, 112.0])
    frame = np.full((112, 112, 3), 10, dtype=np.uint8)
    swapped = np.full((112, 112, 3), 200, dtype=np.uint8)

    result = Refacer._restore_original_mouth_if_open(refacer, frame, swapped, face)

    assert np.array_equal(result, swapped)


def test_restore_original_mouth_noop_when_mouth_closed():
    landmarks = _landmarks_106_with_mouth(top_y=60.0, bottom_y=60.0)  # ratio 0
    refacer = _refacer_with_landmark_model(_FakeLandmarkModel(landmarks))
    face = Face()
    face.bbox = np.array([0.0, 0.0, 112.0, 112.0])
    frame = np.full((112, 112, 3), 10, dtype=np.uint8)
    swapped = np.full((112, 112, 3), 200, dtype=np.uint8)

    result = Refacer._restore_original_mouth_if_open(refacer, frame, swapped, face)

    assert np.array_equal(result, swapped)


def test_restore_original_mouth_blends_when_wide_open():
    # Abertura bem acima de OPEN_MOUTH_RATIO_MAX -> alpha satura em 1.0 dentro
    # do polígono: a região da boca deve vir do frame ORIGINAL, não do swap.
    landmarks = _landmarks_106_with_mouth(top_y=40.0, bottom_y=80.0)  # gap grande
    ratio = Refacer._mouth_open_ratio(Refacer, landmarks)
    assert ratio > Refacer.OPEN_MOUTH_RATIO_MAX  # setup do teste: garante saturação

    refacer = _refacer_with_landmark_model(_FakeLandmarkModel(landmarks))
    face = Face()
    face.bbox = np.array([0.0, 0.0, 112.0, 112.0])
    frame = np.full((112, 112, 3), 10, dtype=np.uint8)
    swapped = np.full((112, 112, 3), 200, dtype=np.uint8)

    result = Refacer._restore_original_mouth_if_open(refacer, frame, swapped, face)

    # No centro da boca, o resultado deve estar muito mais perto do frame
    # original (10) do que do swap (200).
    assert result[60, 56, 0] < 100
    # Longe da boca, o swap continua intocado.
    assert result[5, 5, 0] == 200


def test_restore_original_mouth_alpha_is_gradual_not_a_hard_cutoff():
    # Uma abertura logo ACIMA do mínimo deve produzir um blend mais fraco
    # (mais perto do swap) do que uma abertura bem acima do máximo — a
    # transição é gradual, não liga/desliga abrupto (mitiga flicker entre
    # frames consecutivos com aberturas parecidas).
    refacer_min = _refacer_with_landmark_model(None)  # placeholder, substituído abaixo
    face = Face()
    face.bbox = np.array([0.0, 0.0, 112.0, 112.0])
    frame = np.full((112, 112, 3), 10, dtype=np.uint8)
    swapped = np.full((112, 112, 3), 200, dtype=np.uint8)

    landmarks_barely_open = _landmarks_106_with_mouth(top_y=56.5, bottom_y=63.5)  # gap 7 / width 32 = 0.219
    ratio_barely = Refacer._mouth_open_ratio(Refacer, landmarks_barely_open)
    assert Refacer.OPEN_MOUTH_RATIO_MIN < ratio_barely < Refacer.OPEN_MOUTH_RATIO_MAX

    landmarks_wide_open = _landmarks_106_with_mouth(top_y=40.0, bottom_y=80.0)

    result_barely = Refacer._restore_original_mouth_if_open(
        _refacer_with_landmark_model(_FakeLandmarkModel(landmarks_barely_open)), frame, swapped, face,
    )
    result_wide = Refacer._restore_original_mouth_if_open(
        _refacer_with_landmark_model(_FakeLandmarkModel(landmarks_wide_open)), frame, swapped, face,
    )

    # Ambos os casos escurecem o centro da boca (em direção ao frame original,
    # valor 10) partindo do swap (200), mas a abertura maior deve escurecer
    # MAIS (alpha maior, mais perto do original).
    center = (60, 56, 0)
    assert result_wide[center] < result_barely[center] < 200
