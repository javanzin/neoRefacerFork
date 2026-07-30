"""Testa o formato v2 de exportação/importação (centroides por origem + hash
de conteúdo) e o fluxo de continuação incremental (merge_imported_profile),
com foco especial em retrocompatibilidade: perfis v1 (formato antigo, só
centroide final) devem continuar sendo lidos exatamente como antes — ver
PLANO_IDENTITY_EVOLUTIVO.md, "requisito não negociável" na seção de
compatibilidade.
"""
import numpy as np
import pytest

from insightface.app.common import Face

from identity_profile import (
    EMBEDDING_MODEL_ID,
    PROFILE_FORMAT_VERSION,
    _build_profile_from_samples,
    candidate_as_imported_profile,
    export_profile,
    import_profile,
    imported_profile_known_hashes,
    merge_imported_profile,
)


def _full_sample(embedding, source="src.jpg", origin=None, det_score=0.9, thumbnail=None):
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


def _write_v1_npz(path, embedding, name="Pessoa 1", n_samples=5):
    """Grava um .npz manualmente no formato v1 (pré-existente), sem passar
    por export_profile — simula um arquivo exportado por uma versão do app
    anterior a esta sessão, garantindo que o teste não dependa da própria
    implementação nova para validar a retrocompatibilidade com ela.
    """
    np.savez(
        path,
        embedding=np.asarray(embedding, dtype=np.float32),
        bbox=np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32),
        kps=np.zeros((5, 2), dtype=np.float32),
        det_score=np.float32(0.9),
        name=name,
        n_samples=np.int32(n_samples),
        embedding_model=EMBEDDING_MODEL_ID,
        created_at=np.int64(0),
    )


def test_v1_file_still_imports_without_origins(tmp_path):
    path = tmp_path / "legacy.npz"
    _write_v1_npz(path, [1.0, 0.0, 0.0])

    profile = import_profile(str(path))

    assert profile["name"] == "Pessoa 1"
    assert profile["n_samples"] == 5
    assert np.allclose(profile["face"].embedding, [1.0, 0.0, 0.0])
    assert "origins" not in profile
    assert "profile_format_version" not in profile


def test_export_profile_without_samples_stays_v1(tmp_path):
    """Perfil sem "samples" (ex.: importado de um .npz v1 e reexportado sem
    modificação) não deve ganhar profile_format_version=2 sem dados de
    origem reais — ver docstring de export_profile.
    """
    path = tmp_path / "reexported.npz"
    face = Face(bbox=np.array([0.0, 0.0, 10.0, 10.0]), kps=np.zeros((5, 2)), det_score=0.9)
    face.embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    profile = {"name": "Pessoa 1", "face": face, "n_samples": 3}

    export_profile(profile, str(path))
    reloaded = import_profile(str(path))

    assert "origins" not in reloaded


def test_export_then_import_roundtrip_v2_has_origins(tmp_path):
    samples = [
        _full_sample([1.0, 0.0, 0.0], origin="foto1.jpg"),
        _full_sample([0.9, 0.1, 0.0], origin="foto1.jpg"),
        _full_sample([0.0, 1.0, 0.0], origin="video1.mp4"),
    ]
    profile = _build_profile_from_samples(samples, "Pessoa 1")
    path = tmp_path / "v2.npz"

    export_profile(profile, str(path), content_hashes={"foto1.jpg": "hash1", "video1.mp4": "hash2"})
    reloaded = import_profile(str(path))

    assert reloaded["profile_format_version"] == PROFILE_FORMAT_VERSION
    assert {o["origin"] for o in reloaded["origins"]} == {"foto1.jpg", "video1.mp4"}
    hash_by_origin = {o["origin"]: o["content_hash"] for o in reloaded["origins"]}
    assert hash_by_origin["foto1.jpg"] == "hash1"
    assert hash_by_origin["video1.mp4"] == "hash2"
    n_samples_by_origin = {o["origin"]: o["n_samples"] for o in reloaded["origins"]}
    assert n_samples_by_origin["foto1.jpg"] == 2
    assert n_samples_by_origin["video1.mp4"] == 1


def test_export_without_content_hashes_defaults_to_empty_string(tmp_path):
    samples = [_full_sample([1.0, 0.0, 0.0], origin="foto1.jpg")]
    profile = _build_profile_from_samples(samples, "Pessoa 1")
    path = tmp_path / "no_hash.npz"

    export_profile(profile, str(path))
    reloaded = import_profile(str(path))

    assert reloaded["origins"][0]["content_hash"] == ""


def test_imported_profile_known_hashes_v1_returns_empty_set(tmp_path):
    path = tmp_path / "legacy.npz"
    _write_v1_npz(path, [1.0, 0.0, 0.0])
    profile = import_profile(str(path))

    assert imported_profile_known_hashes(profile) == set()


def test_imported_profile_known_hashes_ignores_empty_hash(tmp_path):
    samples = [_full_sample([1.0, 0.0, 0.0], origin="foto1.jpg")]
    profile = _build_profile_from_samples(samples, "Pessoa 1")
    path = tmp_path / "v2.npz"
    export_profile(profile, str(path))  # sem content_hashes -> hash vazio
    reloaded = import_profile(str(path))

    assert imported_profile_known_hashes(reloaded) == set()


def test_merge_imported_profile_raises_on_v1_profile(tmp_path):
    path = tmp_path / "legacy.npz"
    _write_v1_npz(path, [1.0, 0.0, 0.0])
    profile = import_profile(str(path))
    new_samples = [_full_sample([0.0, 1.0, 0.0], origin="novo.jpg")]

    with pytest.raises(ValueError, match="formato antigo"):
        merge_imported_profile(profile, new_samples)


def test_merge_imported_profile_raises_on_empty_new_samples(tmp_path):
    samples = [_full_sample([1.0, 0.0, 0.0], origin="foto1.jpg")]
    profile = _build_profile_from_samples(samples, "Pessoa 1")
    path = tmp_path / "v2.npz"
    export_profile(profile, str(path))
    reloaded = import_profile(str(path))

    with pytest.raises(ValueError, match="Nenhuma amostra nova"):
        merge_imported_profile(reloaded, [])


def test_merge_imported_profile_combines_legacy_and_new_with_equal_weight():
    """Origem legada (v_N) e origem nova devem pesar igual, independente de
    quantas amostras cruas cada uma tinha originalmente — é o requisito
    central do item 6 do plano. Usa 50 amostras legadas E 50 amostras novas
    (mesma ordem de grandeza): se a agregação tratasse cada amostra crua
    como peso individual em vez de por origem, o resultado seria dominado
    pela origem nova (medido: com amostras legadas=50 vs. novas=1 os dois
    modos davam o mesmo resultado e não distinguiam esse bug — ver
    PLANO_IDENTITY_EVOLUTIVO.md / review de segurança desta sessão).
    """
    v1_samples = [_full_sample([1.0, 0.0, 0.0], origin="video_antigo.mp4") for _ in range(50)]
    profile_v1 = _build_profile_from_samples(v1_samples, "Pessoa 1")

    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "v1.npz")
        export_profile(profile_v1, path)
        imported = import_profile(path)

    new_samples = [_full_sample([0.0, 1.0, 0.0], origin="foto_nova.jpg") for _ in range(50)]

    candidate = merge_imported_profile(imported, new_samples)

    # Peso igual entre as duas origens -> candidato fica a meio caminho entre
    # os dois vetores unitários, não dominado por nenhum dos dois lados
    # apesar de ambos terem 50 amostras cruas na mesma origem.
    expected_direction = np.array([1.0, 1.0, 0.0]) / np.linalg.norm([1.0, 1.0, 0.0])
    result = candidate["face"].embedding
    result_normalized = result / np.linalg.norm(result)
    assert np.allclose(result_normalized, expected_direction, atol=1e-3)


def test_merge_imported_profile_new_samples_from_single_origin_dont_dominate():
    """Caracterização explícita do bug corrigido nesta sessão: MUITAS
    amostras novas de UMA ÚNICA origem não devem pesar mais que a origem
    legada só porque há mais frames — a agregação é por origem, não por
    amostra crua.
    """
    v1_samples = [_full_sample([1.0, 0.0, 0.0], origin="video_antigo.mp4") for _ in range(5)]
    profile_v1 = _build_profile_from_samples(v1_samples, "Pessoa 1")

    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "v1.npz")
        export_profile(profile_v1, path)
        imported = import_profile(path)

    # 50 amostras novas, mas todas da MESMA origem -> continua valendo 1
    # origem, não 50 origens.
    new_samples = [_full_sample([0.0, 1.0, 0.0], origin="video_novo.mp4") for _ in range(50)]
    candidate = merge_imported_profile(imported, new_samples)

    expected_direction = np.array([1.0, 1.0, 0.0]) / np.linalg.norm([1.0, 1.0, 0.0])
    result_normalized = candidate["face"].embedding / np.linalg.norm(candidate["face"].embedding)
    assert np.allclose(result_normalized, expected_direction, atol=1e-3)


def test_merge_imported_profile_candidate_reexports_with_both_origins(tmp_path):
    v1_samples = [_full_sample([1.0, 0.0, 0.0], origin="video_antigo.mp4")]
    profile_v1 = _build_profile_from_samples(v1_samples, "Pessoa 1")
    path_v1 = tmp_path / "v1.npz"
    export_profile(profile_v1, str(path_v1), content_hashes={"video_antigo.mp4": "hashA"})
    imported = import_profile(str(path_v1))

    new_samples = [_full_sample([0.0, 1.0, 0.0], origin="foto_nova.jpg")]
    candidate = merge_imported_profile(imported, new_samples)

    path_v2 = tmp_path / "v2.npz"
    export_profile(candidate, str(path_v2), content_hashes={"foto_nova.jpg": "hashB"})
    reloaded = import_profile(str(path_v2))

    origins_by_name = {o["origin"]: o for o in reloaded["origins"]}
    assert set(origins_by_name) == {"video_antigo.mp4", "foto_nova.jpg"}
    assert origins_by_name["video_antigo.mp4"]["content_hash"] == "hashA"
    assert origins_by_name["foto_nova.jpg"]["content_hash"] == "hashB"


def test_merge_imported_profile_n_samples_sums_legacy_and_new():
    v1_samples = [_full_sample([1.0, 0.0, 0.0], origin="a.jpg") for _ in range(3)]
    profile_v1 = _build_profile_from_samples(v1_samples, "Pessoa 1")

    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "v1.npz")
        export_profile(profile_v1, path)
        imported = import_profile(path)

    new_samples = [_full_sample([0.0, 1.0, 0.0], origin="b.jpg") for _ in range(2)]
    candidate = merge_imported_profile(imported, new_samples)

    assert candidate["n_samples"] == 3 + 2


def test_merge_imported_profile_raises_on_empty_origins(tmp_path):
    """v2 corrompido/artesanal com "origins": [] não deve estourar
    np.stack/ValueError sem contexto ao tentar reexportar — merge_imported_profile
    recusa antes de chegar lá.
    """
    fake_profile = {
        "name": "Pessoa 1",
        "face": Face(bbox=np.array([0.0, 0.0, 10.0, 10.0]), kps=np.zeros((5, 2)), det_score=0.9),
        "n_samples": 0,
        "origins": [],
    }
    new_samples = [_full_sample([0.0, 1.0, 0.0], origin="novo.jpg")]

    with pytest.raises(ValueError, match="nenhuma origem"):
        merge_imported_profile(fake_profile, new_samples)


def test_import_profile_rejects_object_dtype_embedding(tmp_path):
    """import_profile deve usar allow_pickle=False — um .npz malicioso com o
    próprio campo "embedding" como dtype=object (só gravável via pickle) deve
    ser rejeitado ao ser lido, não despicklado. allow_pickle é uma flag
    global do np.load, não por chave: um array object SÓ é rejeitado quando
    efetivamente acessado (np.load não falha s ao abrir o arquivo, mesmo com
    outras chaves object presentes e não lidas) — por isso o teste usa
    "embedding", que import_profile sempre lê, em vez de uma chave v2
    opcional que poderia nunca ser acessada num perfil v1.
    """
    path = tmp_path / "malicious.npz"
    np.savez(
        path,
        # dtype=object só é gravável via pickle — simula um .npz malicioso
        # tentando executar código arbitrário ao ser desserializado.
        embedding=np.asarray([1.0, 0.0, 0.0], dtype=object),
        bbox=np.array([0.0, 0.0, 10.0, 10.0], dtype=np.float32),
        kps=np.zeros((5, 2), dtype=np.float32),
        det_score=np.float32(0.9),
        name="Pessoa 1",
        n_samples=np.int32(1),
        embedding_model=EMBEDDING_MODEL_ID,
    )

    with pytest.raises(ValueError, match="inválido ou corrompido"):
        import_profile(str(path))


def test_v2_file_still_contains_v1_keys_readable_without_origins_support(tmp_path):
    """Um consumidor que só conhece o formato v1 (ex.: código legado que lê
    embedding/bbox/kps/det_score/name/n_samples diretamente do .npz, sem
    passar por import_profile) deve continuar funcionando com um arquivo v2
    — nenhuma chave v1 foi removida ou renomeada.
    """
    samples = [_full_sample([1.0, 0.0, 0.0], origin="foto1.jpg")]
    profile = _build_profile_from_samples(samples, "Pessoa 1")
    path = tmp_path / "v2.npz"
    export_profile(profile, str(path))

    raw = np.load(str(path), allow_pickle=False)
    v1_keys = {"embedding", "bbox", "kps", "det_score", "name", "n_samples", "embedding_model", "created_at"}
    assert v1_keys.issubset(set(raw.files))
    assert np.allclose(raw["embedding"], [1.0, 0.0, 0.0], atol=1e-5)


def test_combined_origin_summaries_disambiguates_colliding_names(tmp_path):
    """Origem nova com o mesmo nome (basename) de uma origem legada — ex.:
    arquivo reenviado com o mesmo nome da v_N mas conteúdo diferente
    (recorte/reencode, que o dedup por hash não pega) — não deve colidir e
    duplicar peso da origem no próximo import.
    """
    v1_samples = [_full_sample([1.0, 0.0, 0.0], origin="video.mp4")]
    profile_v1 = _build_profile_from_samples(v1_samples, "Pessoa 1")
    path_v1 = tmp_path / "v1.npz"
    export_profile(profile_v1, str(path_v1))
    imported = import_profile(str(path_v1))

    # Mesmo nome de origem ("video.mp4") que a origem legada já usa.
    new_samples = [_full_sample([0.0, 1.0, 0.0], origin="video.mp4")]
    candidate = merge_imported_profile(imported, new_samples)

    path_v2 = tmp_path / "v2.npz"
    export_profile(candidate, str(path_v2))
    reloaded = import_profile(str(path_v2))

    origin_names = [o["origin"] for o in reloaded["origins"]]
    assert len(origin_names) == len(set(origin_names)), "nomes de origem duplicados no .npz"
    assert len(reloaded["origins"]) == 2


def test_candidate_as_imported_profile_allows_chaining_merge(tmp_path):
    """Regressão do bug de acumulação corrigido nesta sessão: gerar uma
    candidata, depois continuar A PARTIR DELA (não da v_N original) com mais
    mídia nova deve preservar o material das DUAS rodadas — sem isso, a
    segunda rodada substituía silenciosamente a primeira.
    """
    v1_samples = [_full_sample([1.0, 0.0, 0.0], origin="video_antigo.mp4")]
    profile_v1 = _build_profile_from_samples(v1_samples, "Pessoa 1")
    path_v1 = tmp_path / "v1.npz"
    export_profile(profile_v1, str(path_v1))
    imported = import_profile(str(path_v1))

    # Primeira rodada: adiciona fotos_A.
    fotos_a = [_full_sample([0.0, 1.0, 0.0], origin="fotos_A.jpg")]
    candidate_1 = merge_imported_profile(imported, fotos_a)
    assert candidate_1["n_samples"] == 1 + 1

    # Segunda rodada: continua A PARTIR DA CANDIDATA, adiciona fotos_B.
    base_2 = candidate_as_imported_profile(candidate_1)
    assert {o["origin"] for o in base_2["origins"]} == {"video_antigo.mp4", "fotos_A.jpg"}

    fotos_b = [_full_sample([0.0, 0.0, 1.0], origin="fotos_B.jpg")]
    candidate_2 = merge_imported_profile(base_2, fotos_b)

    # As três origens (legada original + fotos_A + fotos_B) devem estar
    # presentes — nenhuma rodada anterior foi descartada.
    assert set(candidate_2["legacy_origins"] and [o["origin"] for o in candidate_2["legacy_origins"]]) == {
        "video_antigo.mp4", "fotos_A.jpg",
    }
    assert candidate_2["samples"][0]["origin"] == "fotos_B.jpg"
    assert candidate_2["n_samples"] == 1 + 1 + 1


# --- Textura de pele transplantada (opt-in, ver app.apply_identity_anchor) ---
#
# _extract_skin_texture recebe um FRAME + kps (não um crop já alinhado) e
# alinha internamente via face_align.norm_crop, com o mesmo template
# arcface_src usado pelo resto do pipeline de swap (ver docstring da função
# para o porquê — um crop de bbox não tem olhos/nariz/boca em posição fixa).
# Os testes abaixo usam kps == _ARCFACE_LANDMARKS_112 (o próprio template) e
# um frame já do tamanho 112x112: nesse caso norm_crop é a transformação
# identidade (a menos de arredondamento de warpAffine), então o crop alinhado
# é ~igual ao frame de entrada — permite testar a máscara/frequência sem
# depender da geometria exata do warp.

def _frame_with_template_kps(fill_value):
    from identity_profile import _ARCFACE_LANDMARKS_112

    frame = np.full((112, 112, 3), fill_value, dtype=np.uint8)
    return frame, _ARCFACE_LANDMARKS_112.copy()


def test_extract_skin_texture_mask_excludes_landmarks_and_borders():
    from identity_profile import _extract_skin_texture, _ARCFACE_LANDMARKS_112

    frame, kps = _frame_with_template_kps(128)
    result = _extract_skin_texture(frame, kps)
    mask = result["mask"]

    assert mask.shape == (112, 112)
    # Cobertura razoável (bochecha/testa), mas nem todo o crop (exclui
    # cantos de fundo/cabelo e os 5 landmarks).
    assert 0.2 < mask.mean() < 0.6
    # Nenhum dos 5 landmarks (olhos, nariz, boca) fica dentro da máscara.
    for lx, ly in _ARCFACE_LANDMARKS_112:
        assert mask[int(ly), int(lx)] == 0.0
    # Um ponto de bochecha (fora dos raios de exclusão + margem + rampa de
    # feather) deve estar totalmente coberto.
    assert mask[76, 34] == pytest.approx(1.0, abs=1e-4)


def test_skin_region_mask_has_smooth_feathered_edges():
    # A máscara não pode ser binária: uma borda 0→1 em 1px vira um contorno
    # circular visível quando a textura é somada ao frame. Deve existir uma
    # rampa de valores intermediários ao redor de cada exclusão.
    from identity_profile import (
        _SKIN_REGION_MASK_112, _SKIN_TEXTURE_MASK_FEATHER_PX,
        _ARCFACE_LANDMARKS_112, _SKIN_TEXTURE_EYE_EXCLUSION_AXES,
    )

    mask = _SKIN_REGION_MASK_112
    # Fração relevante de pixels com valor fracionário (nem 0 nem 1) — uma
    # máscara binária teria zero.
    fractional = ((mask > 0.0) & (mask < 1.0)).mean()
    assert fractional > 0.05

    # Rampa radial saindo do olho esquerdo para a esquerda, ao longo do EIXO
    # X da elipse de exclusão (mesma linha do centro do landmark, sem offset
    # vertical — longe das outras exclusões e da borda externa, para isolar
    # só essa rampa): zero estrito bem dentro do raio (o blur sobre máscara
    # binária ainda garante isso perto do centro do landmark — só a
    # VIZINHANÇA da borda ganha uma rampa, não a região toda: ver docstring
    # de _skin_region_mask sobre por que uma margem extra antes do blur foi
    # tentada e descartada), subindo a valores altos ao se afastar do centro
    # do landmark, decrescente na direção certa.
    lx, ly = _ARCFACE_LANDMARKS_112[0]
    radius = _SKIN_TEXTURE_EYE_EXCLUSION_AXES[0]  # eixo X da elipse do olho
    row = int(round(ly))
    feather = _SKIN_TEXTURE_MASK_FEATHER_PX

    # Amostra pontos individuais a distâncias crescentes do centro do
    # landmark, ao longo desta linha (para a esquerda, longe de qualquer
    # outra exclusão): deve haver zero estrito bem dentro do raio, uma rampa
    # monotonicamente crescente saindo do raio, e um platô alto uma vez
    # afastado o suficiente (na bochecha).
    def value_at(distance_from_landmark):
        x = int(round(lx - distance_from_landmark))
        return mask[row, x]

    assert value_at(radius * 0.3) == 0.0
    ramp_samples = [value_at(radius + d) for d in np.arange(-feather, feather * 1.5, feather / 4.0)]
    assert np.all(np.diff(ramp_samples) >= -1e-6)
    assert ramp_samples[0] < 0.1
    assert ramp_samples[-1] > 0.9
    # Existe ao menos um valor estritamente intermediário na rampa.
    assert any(0.0 < v < 1.0 for v in ramp_samples)


def test_extract_skin_texture_isolates_medium_band_only():
    from identity_profile import _extract_skin_texture

    # Frame liso (sem textura nenhuma): a banda extraída deve ser ~zero.
    flat, kps = _frame_with_template_kps(128)
    flat_texture = _extract_skin_texture(flat, kps)["texture"]
    # Textura agora é um delta de LUMINÂNCIA (2D), não por canal BGR.
    assert flat_texture.shape == (112, 112)
    assert np.abs(flat_texture).max() < 1.0

    # Ruído somado: deve aparecer na textura extraída (atenuado pelo colapso
    # para luminância e pelo corte de altíssima frequência do band-pass, por
    # isso o limiar é menor que o desvio do ruído de entrada).
    rng = np.random.default_rng(42)
    noisy = np.clip(128 + rng.normal(scale=20, size=(112, 112, 3)), 0, 255).astype(np.uint8)
    noisy_result = _extract_skin_texture(noisy, kps)
    noisy_texture, mask = noisy_result["texture"], noisy_result["mask"]
    assert noisy_texture[mask > 0].std() > 2.0
    # Fora da máscara de pele, a textura é sempre zero (nunca escapa a
    # região de olhos/nariz/boca/fundo).
    assert np.abs(noisy_texture[mask == 0]).max() == 0.0


def test_extract_skin_texture_ignores_chroma_only_variation():
    # Padrão de CROMA puro (canais variam em direções opostas mantendo a
    # luminância constante): não é textura de pele, é cor — não pode vazar
    # para a textura extraída (era a causa da "mancha de tonalidade" quando a
    # extração era feita por canal BGR).
    from identity_profile import _extract_skin_texture

    _, kps = _frame_with_template_kps(128)
    rng = np.random.default_rng(7)
    delta = rng.normal(scale=20, size=(112, 112))
    frame = np.empty((112, 112, 3), dtype=np.float64)
    # Pesos BGR2GRAY: 0.114*dB + 0.587*dG + 0.299*dR == 0 (luminância fixa).
    frame[:, :, 0] = 128 + delta                      # B
    frame[:, :, 1] = 128                              # G
    frame[:, :, 2] = 128 - delta * (0.114 / 0.299)    # R
    frame = np.clip(frame, 0, 255).astype(np.uint8)

    texture = _extract_skin_texture(frame, kps)["texture"]
    # Sobra só o erro de quantização do uint8 (< ~1 nível de cinza).
    assert np.abs(texture).max() < 2.0


def test_apply_skin_texture_never_changes_chroma_of_destination():
    # A textura em luminância deve ser somada IGUALMENTE aos 3 canais do
    # frame de destino — o delta por canal tem que ser idêntico onde não há
    # clipping, preservando a cor original do destino.
    from identity_profile import _extract_skin_texture
    from refacer import Refacer
    from insightface.app.common import Face

    frame, kps = _frame_with_template_kps(128)
    rng = np.random.default_rng(11)
    textured = np.clip(
        128 + rng.normal(scale=15, size=(112, 112, 3)), 0, 255
    ).astype(np.uint8)
    skin_texture = {**_extract_skin_texture(textured, kps), "intensity": 0.5}

    face = Face()
    face.kps = kps
    # Frame de destino com cor uniforme bem longe do neutro (canais bem
    # separados, todos longe de 0/255 para não haver clipping).
    swapped = np.empty((112, 112, 3), dtype=np.uint8)
    swapped[:, :, 0] = 60
    swapped[:, :, 1] = 120
    swapped[:, :, 2] = 180

    result = Refacer._apply_skin_texture(None, swapped, face, skin_texture)

    delta = result.astype(np.int32) - swapped.astype(np.int32)
    # Delta igual nos 3 canais (sem clipping possível com base 60-180 e
    # textura de amplitude pequena) — nenhuma mudança de crominância. A
    # tolerância de 1 nível cobre só o arredondamento float32/uint8 (o mesmo
    # delta somado a bases de magnitude diferente pode truncar 1 nível
    # diferente); vazamento de cor real (o bug corrigido) era de dezenas de
    # níveis.
    assert np.abs(delta[:, :, 0] - delta[:, :, 1]).max() <= 1
    assert np.abs(delta[:, :, 1] - delta[:, :, 2]).max() <= 1
    # E a textura de fato foi aplicada (delta não é zero em toda parte).
    assert np.abs(delta).max() > 0


def test_apply_skin_texture_accepts_legacy_3channel_profiles():
    # Perfis exportados antes da migração guardam textura por canal BGR
    # (shape (112,112,3)) — devem continuar aplicáveis, colapsados para
    # luminância (delta igual nos 3 canais, sem transplantar cor).
    from identity_profile import _SKIN_REGION_MASK_112
    from refacer import Refacer
    from insightface.app.common import Face
    from identity_profile import _ARCFACE_LANDMARKS_112

    rng = np.random.default_rng(13)
    legacy_texture = (
        rng.normal(scale=8, size=(112, 112, 3)).astype(np.float32)
        * _SKIN_REGION_MASK_112[:, :, np.newaxis]
    )
    skin_texture = {
        "texture": legacy_texture,
        "mask": _SKIN_REGION_MASK_112,
        "intensity": 1.0,
    }

    face = Face()
    face.kps = _ARCFACE_LANDMARKS_112.copy()
    swapped = np.full((112, 112, 3), 128, dtype=np.uint8)

    result = Refacer._apply_skin_texture(None, swapped, face, skin_texture)

    assert result.shape == swapped.shape
    delta = result.astype(np.int32) - swapped.astype(np.int32)
    np.testing.assert_array_equal(delta[:, :, 0], delta[:, :, 1])
    np.testing.assert_array_equal(delta[:, :, 1], delta[:, :, 2])
    assert np.abs(delta).max() > 0


def test_skin_texture_roundtrips_through_export_import(tmp_path):
    from identity_profile import _extract_skin_texture

    sample = _full_sample([1.0, 0.0, 0.0])
    profile = _build_profile_from_samples([sample, sample, sample, sample], "Pessoa 1")

    frame, kps = _frame_with_template_kps(100)
    frame[40:60, 40:60] = 180  # alguma variação para não ser puramente liso
    profile["face"].skin_texture = {**_extract_skin_texture(frame, kps), "intensity": 0.7}

    path = tmp_path / "with_skin_texture.npz"
    export_profile(profile, str(path))
    imported = import_profile(str(path))

    assert hasattr(imported["face"], "skin_texture")
    assert imported["face"].skin_texture is not None
    np.testing.assert_allclose(
        imported["face"].skin_texture["texture"], profile["face"].skin_texture["texture"], atol=1e-5,
    )
    np.testing.assert_allclose(
        imported["face"].skin_texture["mask"], profile["face"].skin_texture["mask"],
    )
    assert imported["face"].skin_texture["intensity"] == pytest.approx(0.7)


def test_profile_without_skin_texture_imports_without_the_attribute(tmp_path):
    # Perfil sem textura anexada (o caso comum, opt-in desligado): não deve
    # gravar/ler nenhum campo de skin_texture, nem quebrar o import.
    sample = _full_sample([0.0, 1.0, 0.0])
    profile = _build_profile_from_samples([sample, sample, sample, sample], "Pessoa 1")

    path = tmp_path / "no_skin_texture.npz"
    export_profile(profile, str(path))
    imported = import_profile(str(path))

    assert getattr(imported["face"], "skin_texture", None) is None
