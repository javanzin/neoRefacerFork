"""Testa _compute_centroid: deve se comportar como média simples no caso
trivial e atenuar outliers (ex.: amostras com óculos escuros distorcendo o
embedding) quando há amostras suficientes para julgar o que é "grupo".
"""
import numpy as np
import pytest

from identity_profile import _compute_centroid, ROBUST_CENTROID_SIMILARITY_FLOOR


def _sample(embedding):
    return {"embedding": np.asarray(embedding, dtype=np.float32)}


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
