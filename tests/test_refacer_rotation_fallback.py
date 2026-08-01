"""Testa o fallback de rotação para detecção de rosto em pose "deitada"
(roll ~90°/180°/270° — cabeça inclinada de lado ou de cabeça para baixo,
como alguém deitado numa cama/sofá olhando para a câmera).

Contexto (ver MELHORIAS.md, seção "Pose lateral/deitada"): o alinhamento
(face_align.estimate_norm, SimilarityTransform) já compensa matematicamente
uma rotação pura no plano da imagem — mas só se o detector (SCRFD) primeiro
achar o rosto com keypoints corretos. A hipótese de trabalho (não confirmada
com vídeo real, só por inferência sobre o dataset de treino do SCRFD/
WIDER FACE e por precedente na comunidade — roop-unleashed tem recurso
equivalente, "auto rotation of horizontal faces, fixing bad landmark
positions") é que a rede degrada em confiança/precisão de keypoints quando o
rosto não está "em pé" (roll longe de 0°), mesmo de frente para a câmera —
diferente de yaw/pitch, que é perda real de profundidade 3D e não tem
correção 2D possível.

Mitigação testada aqui: se a detecção no frame original tiver confiança
baixa (ou não achar nada), tentar o frame rotacionado em 90°/180°/270°,
ficar com a rotação de maior score, e desfazer a rotação nas coordenadas
(bbox/kps) antes de retornar — o resto do pipeline nunca precisa saber que
uma rotação aconteceu.

Escopo desta primeira versão (decisão consciente): sem cache/memória da
última rotação "boa" entre frames — testa sempre do zero. Otimização
(lembrar a rotação do frame anterior) fica para depois, se o custo de
CPU/GPU em cenas longas de "deitado" se mostrar um problema real.
"""
import numpy as np
import pytest

from refacer import Refacer


def _bbox_kps(cx, cy, half=20.0, det_score=0.9):
    """Um bbox quadrado simples + 5 keypoints plausíveis (olhos/nariz/boca),
    todos centrados em (cx, cy) — geometria mínima suficiente para os
    testes de rotação de coordenadas, não precisa parecer um rosto real."""
    bbox = np.array([cx - half, cy - half, cx + half, cy + half, det_score], dtype=np.float32)
    kps = np.array([
        [cx - 8, cy - 5],   # olho esquerdo
        [cx + 8, cy - 5],   # olho direito
        [cx, cy],           # nariz
        [cx - 6, cy + 8],   # canto esquerdo da boca
        [cx + 6, cy + 8],   # canto direito da boca
    ], dtype=np.float32)
    return bbox, kps


class _FakeDetector:
    """Substitui self.face_detector nos testes: retorna detecção configurada
    por ângulo de rotação testado (chave = graus de rotação do frame de
    ENTRADA que o detector recebeu), simulando um SCRFD real que só acha o
    rosto (ou só com boa confiança) numa orientação específica."""

    def __init__(self, score_by_angle):
        # score_by_angle: {0: 0.9, 90: 0.2, 180: 0.85, 270: 0.1} etc.
        # Ausência de chave = nenhum rosto detectado nesse ângulo.
        self.score_by_angle = score_by_angle
        self.calls = []

    def detect(self, frame, max_num=0, metric='default'):
        angle = getattr(frame, "_rotation_angle_marker", 0)
        self.calls.append(angle)
        score = self.score_by_angle.get(angle)
        if score is None:
            empty = np.zeros((0, 5), dtype=np.float32)
            return empty, None
        h, w = frame.shape[:2]
        bbox, kps = _bbox_kps(w / 2, h / 2, det_score=score)
        return bbox[np.newaxis, :], kps[np.newaxis, :]


def _marked_frame(shape=(200, 100, 3), angle=0):
    """Frame numpy real (necessário para cv2.rotate de verdade), com um
    atributo extra marcando "que ângulo o _FakeDetector está recebendo" —
    plantado pelo próprio rotate_frame sob teste, não pelo teste diretamente.
    Ver _tag para o mecanismo."""
    return np.zeros(shape, dtype=np.uint8)


@pytest.fixture
def refacer_stub():
    """Refacer sem __init__ real (evita carregar modelos) — só o que os
    métodos de rotação/detecção puros precisam."""
    return Refacer.__new__(Refacer)


class _TaggingRotateFrame:
    """Espiona refacer._rotate_frame para marcar, no array retornado, qual
    ângulo foi aplicado — assim o _FakeDetector sabe "em que orientação" foi
    chamado sem precisar inspecionar pixels."""

    def __init__(self, real_rotate_frame):
        self.real_rotate_frame = real_rotate_frame

    def __call__(self, frame, angle):
        rotated = self.real_rotate_frame(frame, angle)
        tagged = rotated.view()
        tagged._rotation_angle_marker = angle
        return tagged


# numpy views can't carry arbitrary attributes directly on a plain ndarray;
# use a tiny ndarray subclass instead so the marker survives cv2.rotate's
# passthrough in the fake detector path.
class _TaggedArray(np.ndarray):
    _rotation_angle_marker = 0


def _tag(frame, angle):
    tagged = frame.view(_TaggedArray)
    tagged._rotation_angle_marker = angle
    return tagged


def test_uses_frontal_detection_when_score_is_already_good(monkeypatch, refacer_stub):
    """Frame frontal com boa confiança: não deve nem tentar rotacionar
    (comportamento normal, sem custo extra na maioria dos frames)."""
    detector = _FakeDetector({0: 0.9})
    refacer_stub.face_detector = detector
    monkeypatch.setattr(
        "refacer.Refacer._rotate_frame",
        staticmethod(lambda frame, angle: _tag(frame, angle)),
    )

    frame = _tag(_marked_frame(), 0)
    bboxes, kpss = refacer_stub._detect_with_rotation_fallback(frame, max_num=8, metric='default')

    assert detector.calls == [0]
    assert bboxes.shape[0] == 1
    assert bboxes[0, 4] == pytest.approx(0.9)


def test_falls_back_to_180_when_frontal_score_is_low(monkeypatch, refacer_stub):
    """Simula pessoa deitada de cabeça para baixo em relação ao frame
    original: score de 0° é baixo, 180° é bom — deve escolher 180°."""
    detector = _FakeDetector({0: 0.1, 180: 0.88})
    refacer_stub.face_detector = detector
    monkeypatch.setattr(
        "refacer.Refacer._rotate_frame",
        staticmethod(lambda frame, angle: _tag(frame, angle)),
    )

    frame = _tag(_marked_frame(), 0)
    bboxes, kpss = refacer_stub._detect_with_rotation_fallback(frame, max_num=8, metric='default')

    assert 180 in detector.calls
    assert bboxes.shape[0] == 1
    assert bboxes[0, 4] == pytest.approx(0.88)


def test_falls_back_to_best_of_several_candidate_angles(monkeypatch, refacer_stub):
    """Quando mais de uma rotação candidata acha o rosto, fica com a de
    MAIOR det_score, não a primeira encontrada."""
    detector = _FakeDetector({0: 0.15, 90: 0.4, 180: 0.3, 270: 0.6})
    refacer_stub.face_detector = detector
    monkeypatch.setattr(
        "refacer.Refacer._rotate_frame",
        staticmethod(lambda frame, angle: _tag(frame, angle)),
    )

    frame = _tag(_marked_frame(), 0)
    bboxes, kpss = refacer_stub._detect_with_rotation_fallback(frame, max_num=8, metric='default')

    assert bboxes[0, 4] == pytest.approx(0.6)


def test_returns_empty_when_no_rotation_finds_a_face(monkeypatch, refacer_stub):
    """Nenhum ângulo acha rosto: deve retornar vazio, não quebrar."""
    detector = _FakeDetector({})
    refacer_stub.face_detector = detector
    monkeypatch.setattr(
        "refacer.Refacer._rotate_frame",
        staticmethod(lambda frame, angle: _tag(frame, angle)),
    )

    frame = _tag(_marked_frame(), 0)
    bboxes, kpss = refacer_stub._detect_with_rotation_fallback(frame, max_num=8, metric='default')

    assert bboxes.shape[0] == 0
    assert sorted(detector.calls) == [0, 90, 180, 270]


def test_bbox_and_kps_are_unrotated_back_to_original_frame_space(monkeypatch, refacer_stub):
    """O ponto central de teste: bbox/kps retornados devem estar nas
    coordenadas do frame ORIGINAL (não rotacionado) — o resto do pipeline
    (swap, matching, blend) não deve precisar saber que houve rotação.

    Frame original 100x200 (w x h). Rotacionar 90° CW dá um frame 200x100.
    Um rosto detectado no centro do frame rotacionado (100, 50) corresponde,
    desfazendo a rotação, ao centro do frame original (50, 100) — o centro
    geométrico é invariante à rotação, então este é um teste de sanidade
    barato que não exige reimplementar a trigonometria da rotação inversa
    aqui no teste.
    """
    orig_w, orig_h = 100, 200
    detector = _FakeDetector({90: 0.9})
    refacer_stub.face_detector = detector

    def _fake_rotate_90_swaps_shape(frame, angle):
        # 90°/270° de verdade trocam largura<->altura (cv2.rotate real faz
        # isso) — o mock precisa refletir isso para o _FakeDetector "ver" um
        # frame com as dimensões certas ao calcular o centro detectado.
        if angle in (90, 270):
            h, w = frame.shape[:2]
            frame = np.zeros((w, h) + frame.shape[2:], dtype=frame.dtype)
        return _tag(frame, angle)

    monkeypatch.setattr(
        "refacer.Refacer._rotate_frame",
        staticmethod(_fake_rotate_90_swaps_shape),
    )

    frame = _tag(_marked_frame(shape=(orig_h, orig_w, 3)), 0)
    bboxes, kpss = refacer_stub._detect_with_rotation_fallback(frame, max_num=8, metric='default')

    assert bboxes.shape[0] == 1
    cx = (bboxes[0, 0] + bboxes[0, 2]) / 2
    cy = (bboxes[0, 1] + bboxes[0, 3]) / 2
    assert cx == pytest.approx(orig_w / 2, abs=1.0)
    assert cy == pytest.approx(orig_h / 2, abs=1.0)
    # Os 5 keypoints não são perfeitamente simétricos ao redor do centro
    # (boca com offset y maior que os olhos, ver _bbox_kps) — tolerância
    # maior aqui só reflete essa assimetria de fixture, não imprecisão da
    # rotação inversa (já verificada com exatidão no bbox acima).
    assert kpss is not None
    kps_cx = kpss[0][:, 0].mean()
    kps_cy = kpss[0][:, 1].mean()
    assert kps_cx == pytest.approx(orig_w / 2, abs=2.0)
    assert kps_cy == pytest.approx(orig_h / 2, abs=2.0)


def test_rotate_frame_and_unrotate_point_are_mutually_consistent(refacer_stub):
    """Testa _rotate_frame (via cv2.rotate real, não mockado) e
    _unrotate_point juntos: um ponto perto de um canto do frame original,
    levado para o frame rotacionado e detectado lá, deve desfazer para
    perto do canto original correspondente — sem depender do _FakeDetector
    ou de qualquer mock de rotação."""
    orig_h, orig_w = 40, 20
    px, py = 2, 3  # perto do canto superior esquerdo do frame original
    marker_pixel = np.array([255, 1, 1], dtype=np.uint8)  # cor única, sem colisão módulo 256

    for angle in (90, 180, 270):
        frame = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        frame[py, px] = marker_pixel

        rotated = Refacer._rotate_frame(frame, angle)
        rot_h, rot_w = rotated.shape[:2]
        if angle in (90, 270):
            assert (rot_h, rot_w) == (orig_w, orig_h)
        else:
            assert (rot_h, rot_w) == (orig_h, orig_w)

        # A cor (única) do pixel original em (px, py) deve reaparecer em
        # exatamente um ponto do frame rotacionado (rotação é uma permutação
        # de pixels, não interpolação) — usamos isso para achar onde (px, py)
        # foi parar sem reimplementar a trigonometria da rotação direta aqui
        # no teste.
        matches = np.argwhere(np.all(rotated == marker_pixel, axis=-1))
        assert matches.shape[0] == 1
        rot_y, rot_x = matches[0]

        back_x, back_y = refacer_stub._unrotate_point(rot_x, rot_y, angle, orig_w, orig_h)
        assert back_x == pytest.approx(px, abs=1)
        assert back_y == pytest.approx(py, abs=1)


def test_disabled_by_feature_flag_skips_rotation_entirely(monkeypatch, refacer_stub):
    """Espelha o padrão OPEN_MOUTH_FIX_ENABLED: com a flag desligada, chama
    o detector uma única vez em 0° e não tenta nenhuma rotação, mesmo com
    score baixo — reversão total e barata, sem precisar mexer em mais nada.
    """
    import refacer as refacer_module
    monkeypatch.setattr(refacer_module, "ROTATE_FACE_FALLBACK_ENABLED", False)

    detector = _FakeDetector({0: 0.1})
    refacer_stub.face_detector = detector
    monkeypatch.setattr(
        "refacer.Refacer._rotate_frame",
        staticmethod(lambda frame, angle: _tag(frame, angle)),
    )

    frame = _tag(_marked_frame(), 0)
    bboxes, kpss = refacer_stub._detect_with_rotation_fallback(frame, max_num=8, metric='default')

    assert detector.calls == [0]
    assert bboxes.shape[0] == 1
    assert bboxes[0, 4] == pytest.approx(0.1)
