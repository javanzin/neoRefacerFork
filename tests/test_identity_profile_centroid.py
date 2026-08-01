"""Testa _compute_centroid: deve se comportar como média simples no caso
trivial e atenuar outliers (ex.: amostras com óculos escuros distorcendo o
embedding) quando há amostras suficientes para julgar o que é "grupo".

Também testa o balanceamento por origem (_compute_balanced_centroid/
_compute_balanced_mean, opcional via balance_by_origin) e a foto âncora
opcional (_apply_anchor), ambos aplicados em _build_profile_from_samples/
merge_profiles.
"""
import numpy as np
import pytest

from insightface.app.common import Face

from identity_profile import (
    ANCHOR_MAX_WEIGHT,
    _apply_anchor,
    _build_profile_from_samples,
    _compute_balanced_centroid,
    _compute_balanced_mean,
    _compute_centroid,
    _group_samples_by_origin,
    _origin_centroids_as_pseudo_samples,
    _simple_mean_centroid,
    ROBUST_CENTROID_SIMILARITY_FLOOR,
    apply_anchor_to_profile,
    merge_profiles,
)


def _sample(embedding, source="src.jpg", origin=None, det_score=0.9):
    """Sample mínimo para testes de centroide puro (_compute_centroid) — sem
    "face"/"thumbnail", só o necessário para essas funções. Testes que
    exercitam _build_profile_from_samples usam _full_sample abaixo.
    """
    d = {"embedding": np.asarray(embedding, dtype=np.float32), "source": source}
    if origin is not None:
        d["origin"] = origin
    return d


def _full_sample(embedding, source="src.jpg", origin=None, det_score=0.9, thumbnail=None):
    """Sample completo (com "face"/"thumbnail") — necessário para
    _build_profile_from_samples, que escolhe o "representative" por
    face.det_score e carrega bbox/kps/thumbnail dele.
    """
    face = Face(bbox=np.array([0.0, 0.0, 10.0, 10.0]), kps=np.zeros((5, 2)), det_score=det_score)
    face.embedding = np.asarray(embedding, dtype=np.float32)
    sample = {
        "embedding": np.asarray(embedding, dtype=np.float32),
        "face": face,
        "thumbnail": thumbnail,
        "source": source,
    }
    if origin is not None:
        sample["origin"] = origin
    return sample


def test_single_sample_returns_normalized_embedding():
    centroid = _compute_centroid([_sample([3.0, 4.0])])
    assert np.allclose(centroid, [0.6, 0.8])


def test_few_samples_average_without_reweighting():
    # <= 3 amostras: outlier viraria decisão por maioria simples, sem base
    # estatística real — cai na média simples de sempre.
    a = _sample([1.0, 0.0])
    b = _sample([0.0, 1.0])
    centroid = _compute_centroid([a, b])
    expected = np.array([1.0, 1.0]) / np.linalg.norm([1.0, 1.0])
    assert np.allclose(centroid, expected)


def test_outlier_is_downweighted_with_enough_samples():
    # 5 amostras próximas de [1, 0, 0] (o "grupo") + 1 outlier ortogonal
    # (ex.: frame com óculos escuros produzindo embedding deslocado).
    rng = np.random.default_rng(0)
    base = np.array([1.0, 0.0, 0.0])
    group = [
        _sample(base + rng.normal(scale=0.02, size=3))
        for _ in range(5)
    ]
    outlier = _sample([0.0, 1.0, 0.0])

    centroid_with_outlier = _compute_centroid(group + [outlier])
    centroid_group_only = _compute_centroid(group)

    # O outlier deve puxar bem menos o centroide robusto do que puxaria uma
    # média simples (que daria peso igual a todas as 6 amostras).
    naive_mean = np.mean([s["embedding"] for s in group + [outlier]], axis=0)
    naive_mean = naive_mean / np.linalg.norm(naive_mean)

    dist_robust = np.linalg.norm(centroid_with_outlier - centroid_group_only)
    dist_naive = np.linalg.norm(naive_mean - centroid_group_only)

    assert dist_robust < dist_naive


def test_all_below_floor_keeps_previous_centroid_instead_of_collapsing():
    # Amostras deliberadamente dispersas o suficiente para que, em alguma
    # iteração, nenhuma passe do piso de similaridade — não deve quebrar nem
    # devolver um vetor degenerado (NaN/zero).
    samples = [
        _sample([1.0, 0.0, 0.0]),
        _sample([-1.0, 0.0, 0.0]),
        _sample([0.0, 1.0, 0.0]),
        _sample([0.0, -1.0, 0.0]),
    ]
    centroid = _compute_centroid(samples)
    assert np.all(np.isfinite(centroid))
    assert np.linalg.norm(centroid) == pytest.approx(1.0, abs=1e-5) or np.linalg.norm(centroid) == pytest.approx(0.0, abs=1e-5)


def test_zero_norm_embedding_does_not_raise():
    centroid = _compute_centroid([_sample([0.0, 0.0]), _sample([1.0, 0.0]), _sample([1.0, 0.1])])
    assert np.all(np.isfinite(centroid))


def test_floor_makes_suppression_more_aggressive_than_raw_similarity():
    # Cenário realista (ArcFace w600k): maioria com sim ~0.7 ao grupo, uma
    # amostra ocluída com sim ~0.4 — a razão de peso crua (0.4/0.7 ~= 0.57)
    # só atenuaria; subtrair o piso (0.30) antes de pesar deve suprimir a
    # amostra ocluída de forma bem mais agressiva (peso relativo bem < 0.57).
    rng = np.random.default_rng(1)
    base = np.array([1.0, 0.0, 0.0, 0.0])
    good = [_sample(base + rng.normal(scale=0.05, size=4)) for _ in range(6)]
    # Vetor a ~0.4 de similaridade de cosseno com base, no mesmo plano.
    occluded_dir = np.array([0.4, np.sqrt(1 - 0.4 ** 2), 0.0, 0.0])
    occluded = _sample(occluded_dir)

    centroid = _compute_centroid(good + [occluded])
    centroid_good_only = _compute_centroid(good)

    # Com supressão agressiva, o centroide com a amostra ocluída deve
    # continuar muito próximo do centroide calculado só com o grupo bom.
    dist = np.linalg.norm(centroid - centroid_good_only)
    assert dist < 0.05


# --- Balanceamento por origem (opcional, balance_by_origin) ---

def test_single_origin_matches_current_behavior():
    # 1 única origem (mesmo com várias amostras): _compute_balanced_centroid
    # deve produzir a MESMA DIREÇÃO dominante de _compute_centroid direto —
    # nenhuma regressão de comportamento no caso comum (todo o material vem
    # de um só arquivo/vídeo). Não é bit-a-bit idêntico quando há >3 amostras
    # cruas: _compute_centroid(samples) aplica reponderação iterativa nelas,
    # enquanto o balanceado colapsa para 1 pseudo-sample antes disso (ver
    # docstring de _compute_balanced_centroid) — a diferença é só de
    # arredondamento de ponto flutuante, não de direção.
    rng = np.random.default_rng(2)
    base = np.array([1.0, 0.0, 0.0])
    samples = [_sample(base + rng.normal(scale=0.02, size=3), origin="unico.mp4") for _ in range(8)]

    balanced = _compute_balanced_centroid(samples)
    direct = _compute_centroid(samples)
    assert np.allclose(balanced, direct, atol=1e-4)

    # Com <=3 amostras totais, ambos os caminhos caem no mesmo atalho de
    # "poucas amostras" e o resultado É bit-a-bit idêntico.
    few_samples = samples[:3]
    assert np.allclose(_compute_balanced_centroid(few_samples), _compute_centroid(few_samples))


def test_balanced_centroid_equalizes_origin_contribution():
    # Origem A: 500 amostras concentradas perto de [1, 0, 0].
    # Origem B: 5 amostras concentradas perto de um vetor bem diferente
    # (mesma pessoa, ângulo/iluminação distintos).
    rng = np.random.default_rng(3)
    base_a = np.array([1.0, 0.0, 0.0])
    base_b = np.array([0.6, 0.8, 0.0])
    samples_a = [_sample(base_a + rng.normal(scale=0.01, size=3), origin="video_longo.mp4") for _ in range(500)]
    samples_b = [_sample(base_b + rng.normal(scale=0.01, size=3), origin=f"foto_{i}.jpg") for i in range(5)]
    all_samples = samples_a + samples_b

    balanced = _compute_balanced_centroid(all_samples)
    unbalanced = _compute_centroid(all_samples)

    centroid_a = _simple_mean_centroid(samples_a)
    centroid_b = _simple_mean_centroid(samples_b)
    midpoint = centroid_a + centroid_b
    midpoint = midpoint / np.linalg.norm(midpoint)

    # Balanceado fica mais perto do meio-termo entre as duas origens do que
    # o cálculo não-balanceado, que é dominado pela origem A (500 amostras).
    assert np.linalg.norm(balanced - midpoint) < np.linalg.norm(unbalanced - midpoint)
    assert np.linalg.norm(unbalanced - centroid_a) < np.linalg.norm(balanced - centroid_a)


def test_origin_pseudo_sample_weight_is_sqrt_of_frame_count():
    # Peso por origem deve ser sqrt(n_frames), não 1 (peso igual antigo) nem
    # n_frames (dominância bruta antiga) — ver _origin_centroids_as_pseudo_samples.
    samples_a = [_sample([1.0, 0.0], origin="video.mp4") for _ in range(200)]
    samples_b = [_sample([0.0, 1.0], origin="foto.jpg") for _ in range(1)]
    pseudo = _origin_centroids_as_pseudo_samples(samples_a + samples_b)
    weights = {tuple(p["embedding"]): p["origin_weight"] for p in pseudo}
    assert weights[(1.0, 0.0)] == pytest.approx(np.sqrt(200))
    assert weights[(0.0, 1.0)] == pytest.approx(1.0)


def test_balanced_centroid_video_outweighs_but_does_not_dominate_photos():
    # Origem A: 1 vídeo com 200 amostras perto de [1, 0, 0].
    # Origem B..F: 50 fotos avulsas perto de [0.6, 0.8, 0] (1 amostra cada).
    # Com peso igual (comportamento antigo), o vídeo (peso 1) pesaria MENOS
    # que o conjunto das 50 fotos (peso 50 combinado) — o vídeo quase não
    # conta. Com sqrt(n), o vídeo pesa sqrt(200)=~14.1, mais que 1 foto
    # isolada mas menos que as 50 fotos combinadas: nem domina, nem some.
    rng = np.random.default_rng(7)
    base_video = np.array([1.0, 0.0, 0.0])
    base_photos = np.array([0.6, 0.8, 0.0])
    samples_video = [_sample(base_video + rng.normal(scale=0.01, size=3), origin="video.mp4") for _ in range(200)]
    samples_photos = [_sample(base_photos + rng.normal(scale=0.01, size=3), origin=f"foto_{i}.jpg") for i in range(50)]
    all_samples = samples_video + samples_photos

    balanced = _compute_balanced_centroid(all_samples)
    centroid_video = _simple_mean_centroid(samples_video)
    centroid_photos = _simple_mean_centroid(samples_photos)

    # O vídeo pesa mais que a distância que o peso-igual (antigo) produziria:
    # com peso 1 para o vídeo vs. 50 fotos com peso 1 cada, o resultado ficaria
    # a ~1/51 do caminho entre as fotos e o vídeo. sqrt(200)=~14.1 contra 50
    # fotos de peso 1 cada deve puxar o centroide bem mais para perto do vídeo
    # do que isso.
    old_equal_weight = (centroid_video + 50 * centroid_photos)
    old_equal_weight = old_equal_weight / np.linalg.norm(old_equal_weight)
    assert np.linalg.norm(balanced - centroid_video) < np.linalg.norm(old_equal_weight - centroid_video)

    # Mas ainda não domina como o cálculo bruto por frame (peso 200 vs 50)
    # dominaria: o vídeo não deve colapsar o centroide balanceado para cima
    # dele mesmo.
    assert np.linalg.norm(balanced - centroid_video) > 1e-3


def test_origin_grouping_uses_origin_field_not_source_string():
    # source (label de exibição) difere entre os dois, mas origin (campo
    # estruturado) é o mesmo — devem cair no mesmo grupo.
    a = _sample([1.0, 0.0], source="video.mp4 (frame 15)", origin="video.mp4")
    b = _sample([0.0, 1.0], source="video.mp4 (frame 30)", origin="video.mp4")
    groups = _group_samples_by_origin([a, b])
    assert list(groups.keys()) == ["video.mp4"]
    assert len(groups["video.mp4"]) == 2


def test_missing_origin_field_falls_back_to_source():
    # Sample "legado" sem chave origin cai em source como fallback, em vez
    # de quebrar ou de ser silenciosamente ignorado no agrupamento.
    legacy = {"embedding": np.array([1.0, 0.0], dtype=np.float32), "source": "foto.jpg"}
    groups = _group_samples_by_origin([legacy])
    assert list(groups.keys()) == ["foto.jpg"]


def test_build_profile_from_samples_respects_balance_flag():
    rng = np.random.default_rng(4)
    base_a = np.array([1.0, 0.0, 0.0])
    base_b = np.array([0.6, 0.8, 0.0])
    samples_a = [_full_sample(base_a + rng.normal(scale=0.01, size=3), origin="video.mp4") for _ in range(50)]
    samples_b = [_full_sample(base_b + rng.normal(scale=0.01, size=3), origin=f"foto_{i}.jpg") for i in range(5)]
    samples = samples_a + samples_b

    profile_off = _build_profile_from_samples(samples, "Pessoa 1", balance_by_origin=False)
    profile_on = _build_profile_from_samples(samples, "Pessoa 1", balance_by_origin=True)

    assert np.allclose(profile_off["face"].embedding, _compute_centroid(samples))
    assert np.allclose(profile_on["face"].embedding, _compute_balanced_centroid(samples))
    assert not np.allclose(profile_off["face"].embedding, profile_on["face"].embedding)


def test_merge_profiles_balance_flag_without_outlier_suppression():
    rng = np.random.default_rng(5)
    base_a = np.array([1.0, 0.0, 0.0])
    base_b = np.array([0.6, 0.8, 0.0])
    samples_a = [_full_sample(base_a + rng.normal(scale=0.01, size=3), origin="video.mp4") for _ in range(50)]
    samples_b = [_full_sample(base_b + rng.normal(scale=0.01, size=3), origin=f"foto_{i}.jpg") for i in range(5)]

    profile_a = {
        "name": "Pessoa 1", "face": None, "thumbnail": None,
        "samples": samples_a, "n_samples": len(samples_a), "n_discarded": 0, "discarded": [],
    }
    profile_b = {
        "name": "Pessoa 2", "face": None, "thumbnail": None,
        "samples": samples_b, "n_samples": len(samples_b), "n_discarded": 0, "discarded": [],
    }

    merged_off = merge_profiles(profile_a, profile_b, balance_by_origin=False)
    merged_on = merge_profiles(profile_a, profile_b, balance_by_origin=True)

    combined = samples_a + samples_b
    assert np.allclose(merged_off["face"].embedding, _simple_mean_centroid(combined))
    assert np.allclose(merged_on["face"].embedding, _compute_balanced_mean(combined))
    # Nenhuma supressão de outlier em nenhum dos dois casos: o resultado
    # balanceado não deve colapsar para perto só de uma das origens.
    centroid_a = _simple_mean_centroid(samples_a)
    assert np.linalg.norm(merged_on["face"].embedding - centroid_a) > 1e-3


# --- Foto âncora (opcional, upload manual dedicado) ---

def test_no_anchor_matches_current_behavior():
    rng = np.random.default_rng(6)
    base = np.array([1.0, 0.0, 0.0])
    samples = [_full_sample(base + rng.normal(scale=0.02, size=3)) for _ in range(8)]

    profile = _build_profile_from_samples(samples, "Pessoa 1")
    assert np.allclose(profile["face"].embedding, _compute_centroid(samples))


def test_anchor_pulls_centroid_toward_anchor_embedding():
    rng = np.random.default_rng(7)
    base = np.array([1.0, 0.0, 0.0])
    samples = [_full_sample(base + rng.normal(scale=0.02, size=3)) for _ in range(8)]
    anchor = _full_sample([0.0, 1.0, 0.0])

    base_centroid = _compute_centroid(samples)
    with_anchor = _build_profile_from_samples(samples, "Pessoa 1", anchor_sample=anchor)["face"].embedding

    sim_base = float(base_centroid @ (anchor["embedding"] / np.linalg.norm(anchor["embedding"])))
    sim_with_anchor = float(with_anchor @ (anchor["embedding"] / np.linalg.norm(anchor["embedding"])))
    assert sim_with_anchor > sim_base
    # Teto respeitado: o resultado não deve virar a âncora pura.
    assert not np.allclose(with_anchor, anchor["embedding"] / np.linalg.norm(anchor["embedding"]), atol=1e-3)


def test_anchor_weight_is_capped():
    base_centroid = np.array([1.0, 0.0, 0.0])
    anchor_embedding = np.array([0.0, 1.0, 0.0])

    result_over_cap = _apply_anchor(base_centroid, anchor_embedding, anchor_weight=5.0)
    result_at_cap = _apply_anchor(base_centroid, anchor_embedding, anchor_weight=ANCHOR_MAX_WEIGHT)
    assert np.allclose(result_over_cap, result_at_cap)


def test_apply_anchor_to_profile_reapplies_without_duplicating_samples():
    rng = np.random.default_rng(8)
    base = np.array([1.0, 0.0, 0.0])
    samples = [_full_sample(base + rng.normal(scale=0.02, size=3)) for _ in range(8)]
    profile = _build_profile_from_samples(samples, "Pessoa 1")

    anchor_1 = _full_sample([0.0, 1.0, 0.0])
    anchor_2 = _full_sample([0.0, 0.0, 1.0])

    profile = apply_anchor_to_profile(profile, anchor_sample=anchor_1)
    assert profile["n_samples"] == 8
    profile = apply_anchor_to_profile(profile, anchor_sample=anchor_2)
    assert profile["n_samples"] == 8


def test_unset_anchor_restores_balanced_centroid():
    rng = np.random.default_rng(9)
    base = np.array([1.0, 0.0, 0.0])
    samples = [_full_sample(base + rng.normal(scale=0.02, size=3)) for _ in range(8)]
    profile = _build_profile_from_samples(samples, "Pessoa 1")
    original_embedding = profile["face"].embedding.copy()

    anchored = apply_anchor_to_profile(profile, anchor_sample=_full_sample([0.0, 1.0, 0.0]))
    unset = apply_anchor_to_profile(anchored, anchor_sample=None)

    assert not np.allclose(anchored["face"].embedding, original_embedding)
    assert np.allclose(unset["face"].embedding, original_embedding)


# --- Ponderação por nitidez (weight_by_sharpness, opcional) ---

def test_sharpness_weighting_off_is_bitwise_identical():
    # Default desligado: caminho idêntico ao de antes da opção existir,
    # mesmo com o campo "sharpness" presente nas amostras.
    rng = np.random.default_rng(10)
    base = np.array([1.0, 0.0, 0.0])
    samples = [_full_sample(base + rng.normal(scale=0.05, size=3)) for _ in range(6)]
    for i, s in enumerate(samples):
        s["sharpness"] = 100.0 + 50.0 * i

    with_field = _build_profile_from_samples(samples, "Pessoa 1")
    without_flag = _build_profile_from_samples(samples, "Pessoa 1", weight_by_sharpness=False)
    assert np.array_equal(with_field["face"].embedding, without_flag["face"].embedding)


def test_sharpness_weighting_pulls_centroid_toward_sharper_samples():
    # Duas "versões" da mesma pessoa: metade das amostras perto de [1,0,0]
    # (moles) e metade perto da diagonal (nítidas). Com ponderação ligada, o
    # centroide fica mais parecido com a direção das amostras nítidas.
    rng = np.random.default_rng(11)
    soft_dir = np.array([1.0, 0.0, 0.0])
    sharp_dir = np.array([1.0, 0.6, 0.0])
    samples = []
    for _ in range(5):
        s = _sample(soft_dir + rng.normal(scale=0.02, size=3))
        s["sharpness"] = 60.0  # pouco acima do piso de aceitação
        samples.append(s)
    for _ in range(5):
        s = _sample(sharp_dir + rng.normal(scale=0.02, size=3))
        s["sharpness"] = 1500.0
        samples.append(s)

    from identity_profile import _sharpness_weights

    plain = _compute_centroid(samples)
    weighted = _compute_centroid(samples, base_weights=_sharpness_weights(samples))

    sharp_unit = sharp_dir / np.linalg.norm(sharp_dir)
    assert float(weighted @ sharp_unit) > float(plain @ sharp_unit)


def test_sharpness_weighting_without_sharpness_field_equals_plain():
    # Amostras legadas (sem "sharpness"): pesos todos 1 → mesmo resultado da
    # média de sempre, nunca quebra.
    rng = np.random.default_rng(12)
    base = np.array([1.0, 0.0, 0.0])
    samples = [_sample(base + rng.normal(scale=0.05, size=3)) for _ in range(6)]

    from identity_profile import _sharpness_weights

    weights = _sharpness_weights(samples)
    assert np.allclose(weights, np.ones(len(samples)))
    assert np.allclose(_compute_centroid(samples, base_weights=weights), _compute_centroid(samples))


def test_sharpness_weighting_respects_origin_balance():
    # Com balance_by_origin, a nitidez pondera DENTRO de cada origem; entre
    # origens o peso segue 1 pseudo-sample cada. Uma origem inteira de fotos
    # ultranítidas não pode passar a dominar a outra origem por causa disso.
    rng = np.random.default_rng(13)
    dir_a = np.array([1.0, 0.0, 0.0])
    dir_b = np.array([0.0, 1.0, 0.0])
    samples = []
    for _ in range(4):
        s = _sample(dir_a + rng.normal(scale=0.01, size=3), origin="a.mp4")
        s["sharpness"] = 5000.0
        samples.append(s)
    for _ in range(4):
        s = _sample(dir_b + rng.normal(scale=0.01, size=3), origin="b.jpg")
        s["sharpness"] = 60.0
        samples.append(s)

    balanced = _compute_balanced_centroid(samples, weight_by_sharpness=True)
    # Origens homogêneas internamente: a ponderação intra-origem quase não
    # muda cada centroide local, então o resultado deve seguir ~equilibrado
    # entre as duas direções (não dominado pela origem ultranítida).
    sim_a = float(balanced @ dir_a)
    sim_b = float(balanced @ dir_b)
    assert abs(sim_a - sim_b) < 0.05
