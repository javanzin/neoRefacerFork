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
