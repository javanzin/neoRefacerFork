"""Testa cluster_samples(): a decisão de pertencimento a um cluster deve usar
a média simples das amostras do cluster, não o centroide robusto de
_compute_centroid — caso contrário, uma amostra com oclusão (óculos escuros,
por exemplo) perde peso no centroide assim que entra no cluster, o que afasta
esse centroide da aparência "com oclusão" e pode fazer um frame seguinte com
a mesma oclusão falhar o threshold e abrir um cluster novo (a pessoa "vira
duas" mesmo sendo a mesma pessoa ao longo do vídeo).
"""
import numpy as np

from identity_profile import IdentityProfileBuilder, CLUSTER_SIMILARITY_THRESHOLD


class _FakeRecognizer:
    """compute_sim = similaridade de cosseno pura, como o ArcFaceONNX real."""

    def compute_sim(self, a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))


def _sample(embedding):
    return {"embedding": np.asarray(embedding, dtype=np.float32)}


def _builder_with_samples(samples):
    builder = IdentityProfileBuilder(detector=None, recognizer=_FakeRecognizer())
    builder.samples = samples
    return builder


def test_occluded_samples_stay_in_same_cluster_as_group():
    # "Grupo" sem oclusão perto de [1, 0, 0, 0] + várias amostras "com
    # óculos" deslocadas na mesma direção (oclusão real desloca o embedding
    # de forma correlacionada, não em ruído isotrópico aleatório). O shift
    # usado aqui (3.0) coloca a similaridade base-vs-ocluída em ~0.316, logo
    # abaixo de CLUSTER_SIMILARITY_THRESHOLD (0.32) — deliberadamente no
    # limite: é o cenário em que, usando o centroide ROBUSTO na atribuição
    # (o bug que este teste previne), a supressão de peso das amostras
    # ocluídas empurra a comparação da próxima amostra ocluída abaixo do
    # threshold e fragmenta o cluster. Com a média simples usada por
    # cluster_samples(), a comparação continua no grupo. 10+10 amostras (não
    # 5+4) para consolidar o centroide antes da oclusão testar o limite.
    rng = np.random.default_rng(5)
    base = np.array([1.0, 0.0, 0.0, 0.0])
    occlusion_shift = np.array([0.0, 3.0, 0.0, 0.0])

    clean_samples = [
        _sample(base + rng.normal(scale=0.02, size=4)) for _ in range(10)
    ]
    occluded_samples = [
        _sample(base + occlusion_shift + rng.normal(scale=0.02, size=4))
        for _ in range(6)
    ]

    # Amostras limpas primeiro, depois todas as ocluídas em sequência — o
    # cenário de regressão exigia que as ÚLTIMAS amostras com óculos abrissem
    # cluster novo por causa do centroide já supresso pelas anteriores.
    builder = _builder_with_samples(clean_samples + occluded_samples)
    groups = builder.cluster_samples(threshold=CLUSTER_SIMILARITY_THRESHOLD)

    assert len(groups) == 1
    assert len(groups[0]) == len(clean_samples) + len(occluded_samples)


def test_different_people_still_split_into_separate_clusters():
    # Sanity check: a mudança não deve fazer duas pessoas diferentes caírem
    # no mesmo cluster (o problema original que motivou o threshold de 0.32).
    person_a = [_sample([1.0, 0.0, 0.0, 0.0]) for _ in range(3)]
    person_b = [_sample([0.0, 1.0, 0.0, 0.0]) for _ in range(3)]

    builder = _builder_with_samples(person_a + person_b)
    groups = builder.cluster_samples(threshold=CLUSTER_SIMILARITY_THRESHOLD)

    assert len(groups) == 2
    sizes = sorted(len(g) for g in groups)
    assert sizes == [3, 3]
