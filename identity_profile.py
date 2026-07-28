"""Perfil de identidade facial reutilizável, extraído de múltiplas imagens.

Reusa o mesmo detector (SCRFD) e o mesmo extrator de embedding (ArcFaceONNX,
buffalo_l/w600k_r50) já carregados por Refacer — nenhum modelo novo é
introduzido. O perfil resultante é um insightface.app.common.Face sintético,
com .embedding igual ao centroide L2-normalizado das amostras válidas, para
ser consumido exatamente como um dest_face extraído de uma única foto (ver
prepare_faces em refacer.py).
"""

import time

import cv2
import numpy as np
from insightface.app.common import Face
from tqdm import tqdm

# Vídeos são amostrados, não decodificados quadro a quadro: um perfil de
# identidade não precisa de toda a densidade temporal do vídeo (rostos entre
# frames vizinhos são quase idênticos), então um passo fixo, mais agressivo
# que o skip_rate de preview (10) usado no pipeline de swap, já dá amostras
# suficientemente diversas com uma fração do custo de decode/detecção.
#
# Sem teto de frames por vídeo (removido o antigo MAX_FRAMES_PER_VIDEO=60):
# um teto fixo cobria só os primeiros ~30s de qualquer vídeo mais longo
# (60 amostras x 15 de stride), descartando ângulos que só aparecem depois
# disso — numa entrevista de vários minutos, praticamente todo o material
# nunca chegava a ser lido. Em vez de um teto artificial, quem controla o
# custo agora é _is_near_duplicate: barato (downscale + diff de pixel) o
# bastante para pular frames quase idênticos ANTES do custo caro de
# detecção/embedding, então trechos parados (pessoa parada olhando pra
# câmera) custam pouco mesmo sem limite de frames, e trechos com movimento
# real (mudança de ângulo) continuam sendo amostrados.
VIDEO_FRAME_STRIDE = 15

# Frame candidato é descartado (sem rodar detecção) se a diferença média de
# pixel para o último frame ACEITO do mesmo vídeo, em escala de cinza e
# reduzido, ficar abaixo deste limiar — valor baixo o bastante para não
# confundir "pessoa parada" com "mudou de ângulo" (uma pequena guinada de
# cabeça já basta pra passar).
NEAR_DUPLICATE_DOWNSCALE_SIZE = (64, 64)
NEAR_DUPLICATE_MEAN_DIFF_THRESHOLD = 2.0

# Identifica o espaço vetorial do embedding para validar compatibilidade na
# importação (refacer.py carrega w600k_r50.onnx do pacote buffalo_l). O
# INSwapper só aceita embeddings desse modelo — misturar espaços vetoriais
# produz saída anormal (confirmado na documentação oficial do insightface).
EMBEDDING_MODEL_ID = "buffalo_l/w600k_r50"

# Versão do formato de exportação do .npz. v1 (implícito, sem esta chave):
# só o centroide final, sem dados por origem — perfis exportados antes desta
# versão continuam sendo lidos normalmente por import_profile (ver seção 5 de
# PLANO_IDENTITY_EVOLUTIVO.md, "requisito não negociável" de retrocompatibilidade).
# v2: adiciona centroides por origem + hash de conteúdo + contagem por origem,
# permitindo importar um perfil já exportado e continuar a extração
# incrementalmente (ver merge_imported_profile) sem reprocessar o material
# original do zero.
PROFILE_FORMAT_VERSION = 2

MIN_DET_SCORE = 0.5
MIN_FACE_AREA_RATIO = 0.01  # bbox precisa cobrir ao menos 1% da área do frame
MIN_SHARPNESS = 60.0  # variância do Laplaciano no crop alinhado

# Rostos pequenos são sempre upscaled para 112x112 antes de embedding/nitidez
# (ver alignment abaixo), então um corte binário único por área descartava
# rostos pequenos "bons" (nítidos, bem detectados) só por serem pequenos. Um
# piso absoluto continua existindo (MIN_FACE_AREA_RATIO_HARD) porque nitidez
# pós-upscale não captura tudo: o ArcFace foi treinado majoritariamente com
# rostos que já nasciam próximos de 112x112, não upscaled agressivamente de
# poucos pixels — abaixo do piso, o embedding tende a ficar pouco
# discriminativo mesmo quando o crop "parece" nítido (upscale pode gerar
# ringing/artefatos de compressão que o Laplaciano lê como borda real). Entre
# o piso e MIN_FACE_AREA_RATIO, o rosto pequeno só é aceito com evidência
# dupla de qualidade (score e nitidez bem acima do mínimo padrão) — uma
# régua mais alta para compensar a menor quantidade de informação disponível.
MIN_FACE_AREA_RATIO_HARD = 0.0025  # abaixo disso, descarta sempre, sem exceção
MIN_DET_SCORE_COMPENSATED = 0.75  # exigido p/ rosto pequeno na faixa intermediária
MIN_SHARPNESS_COMPENSATED = 90.0  # idem, ~1.5x o mínimo padrão

# Threshold para separar pessoas distintas no clustering (não confundir com o
# 0.2 default do slider "Faces By Match" — aquele é deliberadamente permissivo
# para *confirmar* uma identidade já conhecida dentro do mesmo vídeo; aqui o
# objetivo é o oposto, *separar* pessoas diferentes em material arbitrário, o
# que exige um corte mais alto. Ajustável visualmente na etapa de Revisão.
CLUSTER_SIMILARITY_THRESHOLD = 0.32

# Threshold para find_matches_in_files: aqui a identidade já é conhecida (o
# perfil-alvo, confirmado previamente pelo usuário) e o objetivo é *confirmar*
# esse mesmo rosto em material novo, não separar pessoas diferentes — por
# isso mais permissivo que CLUSTER_SIMILARITY_THRESHOLD, no mesmo espírito do
# default do slider "Faces By Match" citado acima.
TARGET_MATCH_SIMILARITY_THRESHOLD = 0.20

# _compute_centroid pondera amostras pela similaridade ao centroide corrente
# ao longo de ROBUST_CENTROID_ITERATIONS passos, começando pela média simples.
# Amostras "estranhas" ao grupo (oclusão por óculos escuros, mão na frente do
# rosto, ângulo extremo, blur que passou pelo filtro de nitidez) puxam o
# centroide para longe da identidade real na média simples; reponderar pela
# similaridade ao centroide já calculado atenua esse puxão sem precisar saber
# *por que* a amostra é diferente. Duas iterações bastam para convergir na
# prática — a terceira mudaria pouco o resultado e dobraria o custo.
ROBUST_CENTROID_ITERATIONS = 2

# Piso subtraído da similaridade antes de virar peso (peso = max(0, sim -
# piso), não a similaridade crua) — reponderação linear direta é branda
# demais para suprimir outlier (sim 0.40 vs. 0.70 só dá razão de peso 0.57,
# atenua mas não neutraliza). Similaridade cosseno intra-pessoa no ArcFace
# w600k tipicamente fica em 0.35-0.8, então o piso fica abaixo da faixa
# normal — a amostra típica não perde peso por ele, mas amostras realmente
# destoantes (a razão de peso é o que sobra depois de subtrair o piso de
# ambas) são muito mais suprimidas do que na ponderação crua. Mais permissivo
# que CLUSTER_SIMILARITY_THRESHOLD (que separa pessoas *diferentes*): aqui
# todas as amostras já são da mesma pessoa por construção (um único
# build_profile/cluster), então o corte só precisa reconhecer outliers
# claros, não fronteiras entre identidades.
ROBUST_CENTROID_SIMILARITY_FLOOR = 0.30

# Peso máximo que uma foto âncora (upload manual, opcional) pode ter na
# combinação final com o centroide já calculado. Interpolação convexa
# (ver _apply_anchor): 0 = âncora sem nenhum efeito, 1 = âncora substitui
# totalmente o centroide (descartaria o benefício de ter várias referências,
# o oposto do que a âncora deveria fazer). ~0.35 desloca o resultado bem
# menos que a distância típica de similaridade intra-pessoa do ArcFace w600k
# (0.35-0.8, mesma faixa citada acima para ROBUST_CENTROID_SIMILARITY_FLOOR)
# — a âncora ganha influência real sem sair da vizinhança de identidade que o
# pipeline já considera normal. Valor de partida qualitativo, não validado
# por experimento; ajustável visualmente via preview_identity_swap.
ANCHOR_MAX_WEIGHT = 0.35


def _face_sharpness(aligned_crop_bgr):
    gray = cv2.cvtColor(aligned_crop_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _downscale_gray(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, NEAR_DUPLICATE_DOWNSCALE_SIZE)


def _is_near_duplicate(frame_bgr, last_accepted_downscaled):
    """Compara `frame_bgr` ao último frame aceito do mesmo vídeo (já reduzido
    e em escala de cinza) por diferença média absoluta de pixel — muito mais
    barato que detecção/embedding, então serve de filtro antes deles.
    `last_accepted_downscaled` é None no primeiro frame candidato (nunca é
    duplicata).
    """
    if last_accepted_downscaled is None:
        return False
    diff = cv2.absdiff(_downscale_gray(frame_bgr), last_accepted_downscaled)
    return diff.mean() < NEAR_DUPLICATE_MEAN_DIFF_THRESHOLD


def _simple_mean_centroid(samples):
    """Média de embeddings individualmente L2-normalizados, renormalizada no
    final — usada como ponto de partida do centroide robusto e diretamente
    por quem precisa da decisão "essa amostra é da mesma pessoa?" sem
    suprimir nenhuma amostra (cluster_samples() e merge_profiles()).
    """
    embeddings = np.stack([s["embedding"] for s in samples])
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = embeddings / norms

    centroid = normalized.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    return centroid / centroid_norm if centroid_norm > 0 else centroid


def _compute_centroid(samples, iterations=ROBUST_CENTROID_ITERATIONS):
    """Centroide robusto: parte da média simples dos embeddings (L2-normalizados
    individualmente) e refina por `iterations` passos, reponderando cada
    amostra por max(0, similaridade_de_cosseno_ao_centroide - piso). Amostras
    mais parecidas com o grupo pesam mais; outliers (ver
    ROBUST_CENTROID_SIMILARITY_FLOOR) pesam ~0 sem ser removidos da lista.

    Usada para o perfil final (build_profile/build_profiles) — não para a
    decisão de clustering nem para merge_profiles(), que usam a média simples
    direto (ver _simple_mean_centroid) para não suprimir amostras que já
    foram confirmadas como da mesma pessoa.
    """
    embeddings = np.stack([s["embedding"] for s in samples])
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = embeddings / norms

    centroid = _simple_mean_centroid(samples)

    if len(samples) <= 3:
        # Com poucas amostras, "outlier" vira decisão por maioria simples
        # (ex.: 2x1) sem base estatística real para distinguir ruído de
        # variação legítima — mantém a média simples de sempre.
        return centroid

    for _ in range(iterations):
        similarities = normalized @ centroid
        weights = np.clip(similarities - ROBUST_CENTROID_SIMILARITY_FLOOR, 0.0, None)

        if not np.any(weights > 0):
            # Todas as amostras ficaram abaixo do piso (grupo muito
            # heterogêneo) — mantém o centroide da iteração anterior em vez
            # de zerar o resultado.
            break

        weighted = (normalized * weights[:, np.newaxis]).sum(axis=0) / weights.sum()
        weighted_norm = np.linalg.norm(weighted)
        centroid = weighted / weighted_norm if weighted_norm > 0 else centroid

    return centroid


def _group_samples_by_origin(samples):
    """Agrupa `samples` pela origem estruturada (`origin`, ver _add_face_candidate)
    em vez de tentar recuperá-la por parsing do `source` formatado para
    exibição (que carrega sufixos como " (frame 123)"). Amostras "legadas"
    sem a chave `origin` (ex. dict construído manualmente por um teste ou por
    código externo) caem de volta em `source` inteiro — nunca quebra, só deixa
    de agrupar corretamente.

    Retorna um dict comum (não OrderedDict — dict já preserva ordem de
    inserção desde Python 3.7), origem -> list[sample], na ordem de primeira
    aparição, só para determinismo em teste/debug.
    """
    groups = {}
    for s in samples:
        groups.setdefault(s.get("origin", s["source"]), []).append(s)
    return groups


def _origin_centroids_as_pseudo_samples(samples):
    """Um "pseudo-sample" (dict só com `embedding`) por origem distinta,
    cada um sendo o centroide simples (sem supressão de outlier) daquela
    origem — reaproveita _simple_mean_centroid em vez de duplicar a lógica de
    normalização. Alimentar isso de volta em _compute_centroid faz a
    supressão de outlier operar sobre origens, não sobre frames individuais.
    """
    groups = _group_samples_by_origin(samples)
    return [{"embedding": _simple_mean_centroid(group_samples)} for group_samples in groups.values()]


def _origin_summaries(samples, content_hashes=None):
    """Um resumo por origem distinta — nome, centroide simples (mesmo cálculo
    de _origin_centroids_as_pseudo_samples) e contagem de amostras — usado
    para persistir centroides por origem no formato de exportação v2 (ver
    export_profile/PROFILE_FORMAT_VERSION).

    content_hashes (opcional): dict origin -> hash de conteúdo (SHA-256 do
    arquivo original, calculado por quem chama — este módulo não lê arquivos
    do disco por conta própria, ver docstring do módulo). Origem sem hash
    conhecido (ex. perfil originado de merge/âncora, sem arquivo de origem
    direto) grava string vazia.

    Retorna list[dict] com "origin", "centroid", "n_samples", "content_hash",
    na ordem de primeira aparição (mesma ordem de _group_samples_by_origin).
    """
    content_hashes = content_hashes or {}
    groups = _group_samples_by_origin(samples)
    return [
        {
            "origin": origin,
            "centroid": _simple_mean_centroid(group_samples),
            "n_samples": len(group_samples),
            "content_hash": content_hashes.get(origin, ""),
        }
        for origin, group_samples in groups.items()
    ]


def _compute_balanced_centroid(samples):
    """Centroide robusto (mesma supressão de outlier de _compute_centroid),
    mas operando sobre um centroide por origem em vez das amostras cruas —
    um vídeo de centenas de frames vira 1 pseudo-sample, equalizando sua
    contribuição à de uma foto avulsa (que já é, por definição, sua própria
    origem com 1 amostra).

    Com uma única origem, o resultado é a MESMA DIREÇÃO dominante de
    _compute_centroid(samples), mas não necessariamente bit-a-bit idêntico:
    com 1 pseudo-sample (a origem única), a função cai no atalho "<=3
    amostras" e retorna a média simples direto, sem as iterações de
    reponderação que _compute_centroid(samples) aplicaria caso houvesse mais
    de 3 amostras cruas — reponderar 1 pseudo-sample contra si mesmo não
    mudaria o resultado de qualquer forma (toda amostra teria peso 1), então
    a única diferença observável é de arredondamento de ponto flutuante
    (ver test_single_origin_matches_current_behavior), não de direção.
    """
    return _compute_centroid(_origin_centroids_as_pseudo_samples(samples))


def _compute_balanced_mean(samples):
    """Variante de _compute_balanced_centroid SEM supressão de outlier —
    usada por merge_profiles, que deliberadamente evita suprimir qualquer
    amostra (ver docstring de merge_profiles). Ainda assim equaliza a
    contribuição por origem antes da média final.
    """
    return _simple_mean_centroid(_origin_centroids_as_pseudo_samples(samples))


def _apply_anchor(base_centroid, anchor_embedding, anchor_weight):
    """Combina `base_centroid` (já calculado) com o embedding de uma foto
    âncora via interpolação convexa: anchor_weight já é diretamente a fração
    máxima de influência da âncora (0 = sem efeito, 1 = substitui
    completamente o centroide base) — não há dois pesos livres para
    normalizar depois. anchor_weight é sempre clampado a
    [0, ANCHOR_MAX_WEIGHT] aqui (defesa em profundidade: a UI já deveria
    limitar o slider a essa faixa, mas o núcleo não deveria confiar só nisso).
    """
    anchor_weight = float(np.clip(anchor_weight, 0.0, ANCHOR_MAX_WEIGHT))
    anchor_norm = np.linalg.norm(anchor_embedding)
    anchor_normalized = anchor_embedding / anchor_norm if anchor_norm > 0 else anchor_embedding

    combined = (1 - anchor_weight) * base_centroid + anchor_weight * anchor_normalized
    combined_norm = np.linalg.norm(combined)
    return combined / combined_norm if combined_norm > 0 else combined


def _build_profile_from_samples(
    samples,
    name,
    discarded=None,
    balance_by_origin=False,
    anchor_sample=None,
    anchor_weight=ANCHOR_MAX_WEIGHT,
):
    """Núcleo puro (sem estado de instância) de build_profile/build_profiles:
    agrega uma lista de amostras num único perfil. Recebe `samples` e
    `discarded` diretamente em vez de um IdentityProfileBuilder inteiro, para
    não precisar de um objeto parcialmente inicializado (via __new__) só para
    montar o perfil de um cluster já separado.

    balance_by_origin (default False, para não mudar o comportamento hoje já
    validado): se True, agrega via _compute_balanced_centroid (1 centroide
    por origem/arquivo antes da combinação final) em vez do _compute_centroid
    direto sobre todas as amostras — evita que um vídeo com muito mais frames
    que as demais origens domine o perfil só por volume bruto. Com False
    (default), o caminho de código é idêntico ao de antes desta opção
    existir.

    anchor_sample (opcional, default None): amostra de uma foto âncora
    (upload manual dedicado, fora do fluxo de extração normal — ver
    build_anchor_sample), aplicada como uma etapa SEPARADA depois do
    centroide (balanceado ou não) já calculado. Com None (default), não tem
    nenhum efeito — nem o cálculo de _apply_anchor roda.
    """
    if not samples:
        raise ValueError("Nenhuma amostra válida para construir o perfil de identidade.")

    centroid = _compute_balanced_centroid(samples) if balance_by_origin else _compute_centroid(samples)
    if anchor_sample is not None:
        centroid = _apply_anchor(centroid, anchor_sample["embedding"], anchor_weight)

    representative = max(samples, key=lambda s: s["face"].det_score)

    profile_face = Face(
        bbox=representative["face"].bbox,
        kps=representative["face"].kps,
        det_score=representative["face"].det_score,
    )
    profile_face.embedding = centroid

    discarded = discarded or []
    return {
        "name": name,
        "face": profile_face,
        "thumbnail": representative["thumbnail"],
        "samples": list(samples),
        "n_samples": len(samples),
        "n_discarded": len(discarded),
        "discarded": list(discarded),
    }


class IdentityProfileBuilder:
    """Extrai amostras de rosto de várias imagens/vídeos e agrega em perfis.

    Amostras de baixa qualidade são descartadas e contabilizadas, nunca
    silenciadas. build_profile() assume que todas as amostras acumuladas são
    da mesma pessoa (um único perfil, sem separação). build_profiles() faz a
    separação automática multi-pessoa via cluster_samples() (clustering
    greedy por similaridade de embedding) antes de gerar um perfil por
    cluster.
    """

    def __init__(self, detector, recognizer):
        """Accepts the detector/recognizer directly (SCRFD.detect-compatible
        and ArcFaceONNX.get/compute_sim-compatible objects) rather than a
        live Refacer instance — decouples this module from Refacer's
        internals (see from_refacer() for the app.py call site) and lets
        tests pass simple fakes exposing only .detect/.get/.compute_sim.
        """
        self._detector = detector
        self._recognizer = recognizer
        self.samples = []  # list of dict: embedding, face, sharpness, source
        self.discarded = []  # list of dict: source, reason

    @classmethod
    def from_refacer(cls, refacer):
        """Convenience constructor for the real app: pulls the already-loaded
        detector/recognizer off a live Refacer instance (refacer.py loads no
        new models for this — see module docstring).
        """
        return cls(refacer.face_detector, refacer.rec_app)

    def add_image(self, frame_bgr, source_label, origin=None):
        """origin (opcional): identificador cru da origem (ex. nome do
        arquivo, sem sufixos como "(frame N)") usado para balanceamento por
        origem (ver _build_profile_from_samples/balance_by_origin). Se
        omitido, a própria imagem é sua origem (source_label já é o nome cru
        neste caso — não há sufixo a remover).
        """
        if frame_bgr is None:
            self.discarded.append({"source": source_label, "reason": "imagem inválida"})
            return

        bboxes, kpss = self._detector.detect(frame_bgr, max_num=1, metric="max")
        if bboxes.shape[0] == 0:
            self.discarded.append({"source": source_label, "reason": "nenhum rosto detectado"})
            return

        bbox = bboxes[0, 0:4]
        det_score = float(bboxes[0, 4])
        kps = kpss[0] if kpss is not None else None
        self._add_face_candidate(frame_bgr, bbox, kps, det_score, source_label, origin=origin if origin is not None else source_label)

    def _add_face_candidate(self, frame_bgr, bbox, kps, det_score, source_label, origin):
        """Núcleo de validação de qualidade + montagem de amostra
        compartilhado entre add_image (1 rosto por frame, o mais proeminente)
        e find_match_in_frame (N rostos por frame, todos os candidatos que
        baterem com um perfil-alvo).

        origin identifica a origem "crua" da amostra (nome do arquivo/vídeo,
        sem os sufixos de exibição que source_label pode carregar como
        "(frame N)"/"(rosto N)") — usado só para balanceamento por origem,
        nunca para exibição.
        """
        if kps is None:
            self.discarded.append({"source": source_label, "reason": "sem landmarks (kps)"})
            return None

        if det_score < MIN_DET_SCORE:
            self.discarded.append({
                "source": source_label,
                "reason": f"confiança de detecção baixa ({det_score:.2f})",
            })
            return None

        frame_area = frame_bgr.shape[0] * frame_bgr.shape[1]
        bbox_area = max(0.0, (bbox[2] - bbox[0])) * max(0.0, (bbox[3] - bbox[1]))
        face_area_ratio = bbox_area / frame_area if frame_area > 0 else 0.0
        if frame_area <= 0 or face_area_ratio < MIN_FACE_AREA_RATIO_HARD:
            self.discarded.append({"source": source_label, "reason": "rosto pequeno demais no quadro"})
            return None

        embedding = self._recognizer.get(frame_bgr, kps)

        aligned = cv2.resize(
            frame_bgr[max(0, int(bbox[1])):int(bbox[3]), max(0, int(bbox[0])):int(bbox[2])],
            (112, 112),
        ) if bbox[3] > bbox[1] and bbox[2] > bbox[0] else None

        if aligned is None:
            # Degenerate bbox — no crop to judge sharpness on or show as a
            # thumbnail. Discard outright instead of falling back to a
            # sharpness value that would always pass the check below.
            self.discarded.append({"source": source_label, "reason": "bbox inválida (sem crop)"})
            return None

        sharpness = _face_sharpness(aligned)
        if sharpness < MIN_SHARPNESS:
            self.discarded.append({
                "source": source_label,
                "reason": f"imagem desfocada (nitidez {sharpness:.0f})",
            })
            return None

        if face_area_ratio < MIN_FACE_AREA_RATIO:
            # Faixa intermediária: rosto pequeno só entra com evidência dupla
            # de qualidade (score e nitidez bem acima do mínimo padrão), já
            # que o upscale pro embedding tem menos margem de erro.
            if det_score < MIN_DET_SCORE_COMPENSATED or sharpness < MIN_SHARPNESS_COMPENSATED:
                self.discarded.append({
                    "source": source_label,
                    "reason": "rosto pequeno sem compensação suficiente de nitidez/confiança",
                })
                return None

        face = Face(bbox=bbox, kps=kps, det_score=det_score)
        face.embedding = embedding

        sample = {
            "embedding": embedding,
            "face": face,
            "thumbnail": aligned,
            "source": source_label,
            "origin": origin,
        }
        self.samples.append(sample)
        return sample

    def add_video(self, video_path, source_label):
        """Amostra frames de um vídeo (passo fixo, sem teto de quadros) e
        alimenta cada um em add_image — mesmo filtro de qualidade das
        imagens, sem decodificar o vídeo inteiro nem mantê-lo todo em memória
        (diferente de analyze_video_in_memory, que existe para o caminho de
        swap, não de extração de identidade).

        Frames quase idênticos ao último frame aceito (ver
        _is_near_duplicate) são pulados antes do custo de detecção/embedding
        — cobre o vídeo inteiro sem gastar esse custo em trechos parados.

        Todos os frames deste vídeo compartilham a mesma `origin` (o
        `source_label` cru recebido aqui, sem o sufixo "(frame N)" que o
        label de exibição ganha) — usado só para balanceamento por origem.
        """
        self._sample_video_frames(
            video_path,
            source_label,
            lambda frame, display_label: self.add_image(frame, display_label, origin=source_label),
        )

    def _sample_video_frames(self, video_path, source_label, on_frame):
        """Núcleo de amostragem de vídeo (stride fixo + filtro de frame quase
        idêntico) compartilhado entre add_video (extrai amostras de todas as
        pessoas) e find_matches_in_video (extrai só as que baterem com um
        perfil-alvo) — só muda o que cada um faz com o frame amostrado.
        """
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            self.discarded.append({"source": source_label, "reason": "vídeo não pôde ser aberto"})
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = 0
        last_accepted_downscaled = None
        n_duplicates = 0

        with tqdm(total=total_frames, desc=f"Extraindo identidade de {source_label}") as pbar:
            while cap.isOpened():
                flag, frame = cap.read()
                if not flag:
                    break

                if frame_index % VIDEO_FRAME_STRIDE != 0:
                    frame_index += 1
                    pbar.update()
                    continue

                if _is_near_duplicate(frame, last_accepted_downscaled):
                    n_duplicates += 1
                    frame_index += 1
                    pbar.update()
                    continue

                last_accepted_downscaled = _downscale_gray(frame)
                on_frame(frame, f"{source_label} (frame {frame_index})")
                frame_index += 1
                pbar.update()

        cap.release()

        if n_duplicates:
            self.discarded.append({
                "source": source_label,
                "reason": f"{n_duplicates} frame{'s' if n_duplicates != 1 else ''} quase idêntico(s) ao anterior, pulado(s)",
            })

    def find_match_in_frame(self, frame_bgr, source_label, target_face, threshold=TARGET_MATCH_SIMILARITY_THRESHOLD, origin=None):
        """Busca dirigida num único frame/imagem já decodificado: em vez de
        extrair e depois separar por pessoa (add_image + cluster_samples),
        procura diretamente por UMA identidade já conhecida (target_face, o
        Face de um perfil já confirmado).

        Detecta TODOS os rostos do frame (max_num=0, ao contrário de
        add_image que só pega o mais proeminente), compara cada um ao
        embedding do alvo e só chama _add_face_candidate (logo, só gasta os
        filtros de qualidade) nos que baterem — descarta os demais sem
        clusterizá-los. Muito mais barato que build_profiles() em material
        com várias pessoas, já que não há comparação O(n²) entre todo mundo:
        cada rosto candidato é comparado a 1 alvo só.

        Retorna a lista de novas amostras aceitas neste frame (mesmo formato
        de self.samples), já também acrescentadas a
        self.samples/self.discarded.

        origin (opcional): mesmo papel de add_image — identificador cru da
        origem, para balanceamento por origem. Se omitido, usa source_label
        (imagem avulsa: ela é sua própria origem).
        """
        target_embedding = target_face.embedding
        matches = []
        origin = origin if origin is not None else source_label

        bboxes, kpss = self._detector.detect(frame_bgr, max_num=0)
        if bboxes.shape[0] == 0:
            self.discarded.append({"source": source_label, "reason": "nenhum rosto detectado"})
            return matches

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = float(bboxes[i, 4])
            kps = kpss[i] if kpss is not None else None
            if kps is None:
                continue

            candidate_embedding = self._recognizer.get(frame_bgr, kps)
            if self._recognizer.compute_sim(target_embedding, candidate_embedding) < threshold:
                continue

            label = f"{source_label} (rosto {i + 1})" if bboxes.shape[0] > 1 else source_label
            sample = self._add_face_candidate(frame_bgr, bbox, kps, det_score, label, origin=origin)
            if sample is not None:
                matches.append(sample)

        if not matches:
            self.discarded.append({"source": source_label, "reason": "nenhum rosto bateu com o perfil-alvo"})

        return matches

    def find_matches_in_video(self, video_path, source_label, target_face, threshold=TARGET_MATCH_SIMILARITY_THRESHOLD):
        """Aplica find_match_in_frame a cada frame amostrado de um vídeo
        (mesmo stride/filtro de quase-duplicata de add_video, via
        _sample_video_frames), em vez de extrair todas as pessoas do vídeo
        para depois separar por similaridade. Todos os frames deste vídeo
        compartilham a mesma origin (o nome cru do vídeo), para balanceamento
        por origem.
        """
        matches = []
        self._sample_video_frames(
            video_path,
            source_label,
            lambda frame, label: matches.extend(
                self.find_match_in_frame(frame, label, target_face, threshold, origin=source_label)
            ),
        )
        return matches

    def build_profile(self, name="Pessoa 1", balance_by_origin=False, anchor_sample=None, anchor_weight=ANCHOR_MAX_WEIGHT):
        return _build_profile_from_samples(
            self.samples, name, self.discarded,
            balance_by_origin=balance_by_origin, anchor_sample=anchor_sample, anchor_weight=anchor_weight,
        )

    def cluster_samples(self, threshold=CLUSTER_SIMILARITY_THRESHOLD):
        """Separa self.samples em grupos por pessoa (greedy, sem lib de
        clustering nova — usa apenas self._recognizer.compute_sim, a mesma
        função já usada em _apply_swaps para matching).

        Cada amostra é atribuída ao cluster de MAIOR similaridade dentre os
        que passam o threshold (não ao primeiro encontrado) — evita que uma
        amostra ruim early contamine um centroide por mero acaso de ordem de
        processamento. Se nenhum cluster existente passa o threshold, uma
        amostra abre um cluster novo.

        A atribuição usa a MÉDIA SIMPLES (incremental, não o centroide
        robusto de _compute_centroid) das amostras do cluster: aqui a
        pergunta é "esta amostra é da mesma pessoa?", não "qual o melhor
        vetor representativo desta pessoa?". Uma amostra com oclusão (óculos
        escuros, por exemplo) ainda é da mesma pessoa e não deve perder peso
        na decisão de pertencimento — se perdesse, o centroide do cluster se
        afastaria da aparência "com oclusão" e um frame seguinte com a mesma
        oclusão poderia falhar o threshold e abrir um cluster novo (a pessoa
        "vira duas"), justamente o sintoma que o centroide robusto deveria
        evitar. O centroide robusto entra depois, uma única vez por grupo já
        fechado, em _build_profile_from_samples/build_profiles.

        Retorna list[list[sample]], na ordem de criação dos clusters (cluster
        0 = "Pessoa 1", etc.) — mera convenção de nomenclatura neutra, nunca
        inferida de metadado de arquivo.
        """
        clusters = []  # list of {"centroid": np.ndarray, "sum": np.ndarray, "samples": [sample, ...]}

        for sample in self.samples:
            emb = sample["embedding"]
            norm = np.linalg.norm(emb)
            normalized_emb = emb / norm if norm > 0 else emb

            best_idx, best_sim = -1, -1.0
            for idx, cluster in enumerate(clusters):
                sim = self._recognizer.compute_sim(cluster["centroid"], emb)
                if sim > best_sim:
                    best_idx, best_sim = idx, sim

            if best_idx >= 0 and best_sim >= threshold:
                cluster = clusters[best_idx]
                cluster["samples"].append(sample)
                # Média incremental simples (O(1) por amostra) dos embeddings
                # normalizados — não o centroide robusto, ver docstring acima.
                cluster["sum"] = cluster["sum"] + normalized_emb
                mean = cluster["sum"] / len(cluster["samples"])
                mean_norm = np.linalg.norm(mean)
                cluster["centroid"] = mean / mean_norm if mean_norm > 0 else mean
            else:
                clusters.append({
                    "centroid": normalized_emb,
                    "sum": normalized_emb.copy(),
                    "samples": [sample],
                })

        return [c["samples"] for c in clusters]

    def build_profiles(self, threshold=CLUSTER_SIMILARITY_THRESHOLD, balance_by_origin=False):
        """Separa as amostras em clusters por pessoa e constrói um perfil
        (centroide + Face sintético) por cluster, nomeados "Pessoa 1",
        "Pessoa 2"... na ordem de criação dos clusters.

        balance_by_origin (default False): repassado a _build_profile_from_samples
        — ver docstring lá para o que muda. Âncora não se aplica aqui: ela é
        escolhida pelo usuário depois de já ver os perfis extraídos (ver
        apply_anchor_to_profile).
        """
        if not self.samples:
            raise ValueError("Nenhuma amostra válida para construir perfis de identidade.")

        groups = self.cluster_samples(threshold=threshold)
        profiles = [
            _build_profile_from_samples(group, name=f"Pessoa {i + 1}", balance_by_origin=balance_by_origin)
            for i, group in enumerate(groups)
        ]

        # Descartes de qualidade (add_image/add_video) pertencem à extração
        # como um todo, não a um cluster específico — anexados apenas ao
        # primeiro perfil para não duplicar a contagem em todos.
        if profiles:
            profiles[0]["n_discarded"] = len(self.discarded)
            profiles[0]["discarded"] = list(self.discarded)

        return profiles


def merge_profiles(profile_a, profile_b, name=None, balance_by_origin=False):
    """Combina dois perfis (tipicamente 'Pessoa X' e 'Pessoa Y' que o
    clustering separou por engano, mas são a mesma pessoa) num único perfil,
    recalculando o centroide a partir da união das amostras de ambos — não é
    uma média dos dois centroides já prontos, que pesaria igualmente um
    cluster com 2 amostras e um com 20.

    Usa média simples (não o centroide robusto de _compute_centroid): o
    merge é uma correção manual — o usuário já olhou as amostras dos dois
    clusters e decidiu que são a mesma pessoa. Um cluster separado por óculos
    escuros é o caso típico que motiva o merge; suprimir essas amostras de
    novo aqui (via reponderação por similaridade) anularia silenciosamente a
    correção que o usuário acabou de pedir. A supressão de outliers já teve
    sua chance em cluster_samples()/build_profiles() antes do usuário decidir
    mesclar.

    balance_by_origin (default False): se True, usa _compute_balanced_mean
    (equaliza a contribuição por origem/arquivo) em vez de _simple_mean_centroid
    puro — mas continua SEM supressão de outlier (_compute_centroid), pela
    mesma razão do parágrafo acima: balancear por origem não é o mesmo que
    suprimir amostras, e o merge não deve fazer a segunda coisa.

    Requer que ambos os perfis tenham a chave "samples" — perfis vindos de
    import_profile() não a têm (o .npz exportado guarda só o centroide, não
    as amostras individuais, por design de privacidade) e não podem ser
    mesclados.
    """
    if "samples" not in profile_a or "samples" not in profile_b:
        raise ValueError(
            "Não é possível mesclar: perfis importados de um arquivo .npz não "
            "retêm as amostras individuais (só o centroide final é exportado). "
            "Mesclagem só funciona entre perfis extraídos na sessão atual."
        )

    combined_samples = profile_a["samples"] + profile_b["samples"]
    representative = max(combined_samples, key=lambda s: s["face"].det_score)

    profile_face = Face(
        bbox=representative["face"].bbox,
        kps=representative["face"].kps,
        det_score=representative["face"].det_score,
    )
    profile_face.embedding = (
        _compute_balanced_mean(combined_samples) if balance_by_origin else _simple_mean_centroid(combined_samples)
    )

    return {
        "name": name or profile_a["name"],
        "face": profile_face,
        "thumbnail": representative["thumbnail"],
        "samples": combined_samples,
        "n_samples": len(combined_samples),
        "n_discarded": profile_a["n_discarded"] + profile_b["n_discarded"],
        "discarded": profile_a["discarded"] + profile_b["discarded"],
    }


def build_anchor_sample(detector, recognizer, frame_bgr, source_label):
    """Processa uma foto âncora (upload manual dedicado, opcional) pelo mesmo
    caminho de qualidade/embedding do resto do pipeline (via um
    IdentityProfileBuilder efêmero, só para reaproveitar _add_face_candidate
    sem duplicar a validação de nitidez/score/tamanho) — uma âncora de má
    qualidade não deve ser aceita silenciosamente.

    Retorna a amostra aceita (dict com "embedding", compatível com
    anchor_sample de apply_anchor_to_profile) ou None se rejeitada, e a lista
    de descartes (mesmo formato de IdentityProfileBuilder.discarded) para o
    chamador reportar o motivo ao usuário.
    """
    builder = IdentityProfileBuilder(detector, recognizer)
    builder.add_image(frame_bgr, source_label)
    anchor_sample = builder.samples[0] if builder.samples else None
    return anchor_sample, builder.discarded


def apply_anchor_to_profile(profile, anchor_sample=None, anchor_weight=ANCHOR_MAX_WEIGHT, balance_by_origin=False):
    """Reaplica (ou remove, com anchor_sample=None) a foto âncora sobre um
    perfil já construído, recalculando a partir de profile["samples"] em vez
    de acumular — chamar duas vezes seguidas com âncoras diferentes não
    duplica amostras nem cresce n_samples.

    balance_by_origin deve refletir a mesma opção usada para construir o
    perfil originalmente (ver _build_profile_from_samples), para que reaplicar
    a âncora não mude silenciosamente essa outra escolha.

    Requer profile["samples"] — perfis importados de .npz não os têm (mesma
    limitação de merge_profiles), então não podem receber uma âncora.
    """
    if "samples" not in profile:
        raise ValueError(
            "Não é possível aplicar âncora: perfis importados de um arquivo .npz não "
            "retêm as amostras individuais. Âncora só funciona em perfis extraídos na sessão atual."
        )

    return _build_profile_from_samples(
        profile["samples"],
        name=profile["name"],
        discarded=profile["discarded"],
        balance_by_origin=balance_by_origin,
        anchor_sample=anchor_sample,
        anchor_weight=anchor_weight,
    )


def export_profile(profile, output_path, content_hashes=None):
    """Grava o perfil em .npz.

    Sempre grava o centroide final (compatível com o formato v1 lido por
    versões antigas do import_profile e por qualquer consumidor externo que
    só olhe embedding/bbox/kps/det_score/name/n_samples — nenhum campo do
    formato v1 foi removido ou renomeado).

    Quando profile["samples"] está disponível (perfil extraído/mesclado na
    sessão atual — perfis vindos de um import_profile não o têm, ver
    merge_profiles), grava ADICIONALMENTE um centroide por origem (nome,
    centroide, n_samples, hash de conteúdo) sob profile_format_version=2 —
    isso é o que permite reimportar o perfil depois e continuar a extração
    incrementalmente (ver merge_imported_profile) sem reprocessar o material
    original. Sem "samples" (perfil já importado sendo reexportado sem
    modificação, ex. só renomeado), grava apenas o formato v1: não há como
    reconstruir centroides por origem sem as amostras, e forçar
    profile_format_version=2 sem dados por origem quebraria a suposição de
    merge_imported_profile de que v2 sempre tem ao menos uma origem.

    content_hashes (opcional): dict origin -> hash de conteúdo (SHA-256 do
    arquivo original — calculado por quem chama, tipicamente app.py reusando
    o mesmo hash já calculado para deduplicação). Origem sem hash conhecido
    grava string vazia (ver _origin_summaries).
    """
    face = profile["face"]
    fields = dict(
        embedding=face.embedding.astype(np.float32),
        bbox=np.asarray(face.bbox, dtype=np.float32),
        kps=np.asarray(face.kps, dtype=np.float32) if face.kps is not None else np.zeros((5, 2), dtype=np.float32),
        det_score=np.float32(face.det_score),
        name=profile["name"],
        n_samples=np.int32(profile["n_samples"]),
        embedding_model=EMBEDDING_MODEL_ID,
        created_at=np.int64(int(time.time())),
    )

    samples = profile.get("samples")
    origin_summaries = None
    if "legacy_origins" in profile:
        origin_summaries = _combined_origin_summaries(profile, content_hashes=content_hashes)
    elif samples:
        origin_summaries = _origin_summaries(samples, content_hashes=content_hashes)

    if origin_summaries:
        # dtype=str (não object): arrays de string de comprimento variável
        # fazem roundtrip completo em .npz com dtype unicode nativo — dtype=object
        # gravaria via pickle, o que forçaria import_profile a usar
        # allow_pickle=True e abriria execução de código arbitrário ao importar
        # um .npz malicioso (allow_pickle é uma flag global do arquivo inteiro,
        # não por chave — qualquer array object no arquivo seria despicklado).
        fields.update(
            profile_format_version=np.int32(PROFILE_FORMAT_VERSION),
            origin_names=np.asarray([s["origin"] for s in origin_summaries], dtype=str),
            origin_centroids=np.stack([s["centroid"].astype(np.float32) for s in origin_summaries]),
            origin_n_samples=np.asarray([s["n_samples"] for s in origin_summaries], dtype=np.int32),
            origin_content_hashes=np.asarray([s["content_hash"] for s in origin_summaries], dtype=str),
        )

    np.savez(output_path, **fields)
    return output_path


def import_profile(npz_path):
    """Carrega um perfil exportado, validando a compatibilidade do espaço vetorial.

    Levanta ValueError (não silencioso) se o arquivo não for um perfil válido
    ou tiver sido gerado por um modelo de embedding diferente.

    Perfis no formato v1 (sem profile_format_version, exportados antes desta
    versão) continuam sendo lidos exatamente como antes — o retorno não tem
    "origins" nesse caso. Perfis v2 ganham a chave adicional "origins"
    (list[dict] com origin/centroid/n_samples/content_hash), consumida por
    merge_imported_profile para continuar a extração incrementalmente.
    """
    try:
        data = np.load(npz_path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"Arquivo de perfil inválido ou corrompido: {exc}") from exc

    required_keys = {"embedding", "bbox", "kps", "det_score", "name", "n_samples", "embedding_model"}
    if not required_keys.issubset(data.files):
        raise ValueError("Arquivo não contém os campos esperados de um perfil de identidade.")

    # allow_pickle=False só rejeita um array dtype=object quando a chave é
    # efetivamente ACESSADA (np.load acima não falha ao abrir o arquivo,
    # mesmo com arrays object presentes e não lidos) — por isso todo acesso
    # aos campos abaixo fica dentro deste try, não só a abertura do arquivo,
    # garantindo que um .npz malicioso sempre produza o ValueError "inválido
    # ou corrompido" tratado, nunca deixe a exceção nativa do numpy escapar
    # sem esse contexto. embedding_model é lido antes deste bloco porque sua
    # incompatibilidade é um erro de NEGÓCIO com mensagem própria, não um
    # arquivo corrompido — não deve ser reclassificado como "corrompido".
    embedding_model = str(data["embedding_model"])
    if embedding_model != EMBEDDING_MODEL_ID:
        raise ValueError(
            f"Perfil incompatível: foi gerado com o modelo de embedding "
            f"'{embedding_model}', mas este app usa '{EMBEDDING_MODEL_ID}'. "
            "Misturar espaços vetoriais diferentes produz swaps incorretos."
        )

    try:
        face = Face(
            bbox=data["bbox"],
            kps=data["kps"],
            det_score=float(data["det_score"]),
        )
        face.embedding = data["embedding"].astype(np.float32)

        profile = {
            "name": str(data["name"]),
            "face": face,
            "n_samples": int(data["n_samples"]),
        }

        origin_keys = {"profile_format_version", "origin_names", "origin_centroids", "origin_n_samples", "origin_content_hashes"}
        if origin_keys.issubset(data.files):
            origin_names = data["origin_names"]
            origin_centroids = data["origin_centroids"]
            origin_n_samples = data["origin_n_samples"]
            origin_content_hashes = data["origin_content_hashes"]
            profile["profile_format_version"] = int(data["profile_format_version"])
            profile["origins"] = [
                {
                    "origin": str(origin_names[i]),
                    "centroid": origin_centroids[i].astype(np.float32),
                    "n_samples": int(origin_n_samples[i]),
                    "content_hash": str(origin_content_hashes[i]),
                }
                for i in range(len(origin_names))
            ]
    except Exception as exc:
        raise ValueError(f"Arquivo de perfil inválido ou corrompido: {exc}") from exc

    return profile


def imported_profile_known_hashes(imported_profile):
    """Hashes de conteúdo já conhecidos de um perfil importado (v2), para
    dedup: arquivo novo cujo hash já aparece aqui é ignorado antes de
    reprocessar (ver PLANO_IDENTITY_EVOLUTIVO.md, item 3 do fluxo). Perfil v1
    (sem "origins") ou origem sem hash conhecido (string vazia, ver
    _origin_summaries) simplesmente não contribui nenhum hash — dedup
    continua funcionando, só não pega esses casos.

    Retorna um set (hash de conteúdo -> nada útil de mapear, só a presença
    importa aqui).
    """
    origins = imported_profile.get("origins", [])
    return {o["content_hash"] for o in origins if o["content_hash"]}


def merge_imported_profile(imported_profile, new_samples, name=None, new_discarded=None):
    """Combina um perfil IMPORTADO (v2, com profile["origins"]) com amostras
    NOVAS (de mídia adicional processada nesta sessão, via
    IdentityProfileBuilder), produzindo uma candidata sem precisar
    reprocessar o material original da v_N (ver PLANO_IDENTITY_EVOLUTIVO.md,
    fluxo de atualização incremental).

    Cada origem legada da v_N entra na combinação como 1 pseudo-sample de
    peso IGUAL às novas origens (não escalado por n_samples) — escalar pelo
    n_samples reintroduziria o viés de volume que o balanceamento por origem
    existe para evitar, já que não há garantia sobre a distribuição interna
    de uma origem legada sem suas amostras individuais preservadas (ver item
    6 do fluxo no plano).

    SEM parâmetro balance_by_origin (diferente de _build_profile_from_samples):
    aqui não existe um modo "sem balanceamento" coerente, porque as origens
    legadas só existem como 1 centroide cada — tratá-las como amostras cruas
    individuais (o que balance_by_origin=False faria) reduziria cada origem
    legada inteira ao peso de 1 frame novo, apagando a identidade da v_N
    quando há muitas amostras novas de uma origem só (medido: 50 amostras
    legadas vs. 50 novas de 1 vídeo produzia um centroide 99,98% dominado
    pelo vídeo novo). Sempre agrega via _compute_centroid sobre pseudo-samples
    por origem (peso igual entre todas), equivalente ao balance_by_origin=True
    de _build_profile_from_samples.

    A candidata resultante:
    - Tem profile["samples"] = só as NOVAS amostras (as legadas nunca tiveram
      samples individuais preservados, apenas o centroide por origem) — isso
      é suficiente para export_profile gravar um v2 novo (via
      _combined_origin_summaries abaixo, não via profile["samples"] direto).
    - NÃO suporta merge_profiles/apply_anchor_to_profile diretamente sobre as
      origens legadas (mesma limitação de sempre: sem amostras individuais).
      Aplicar âncora precisa ser feito ANTES de exportar novamente, sobre as
      novas amostras via apply_anchor_to_profile de um perfil comum, ou
      tratado como uma etapa separada — não coberto por esta função.

    Levanta ValueError se imported_profile não tiver "origins" (perfil v1,
    sem dados por origem — não há nada para continuar incrementalmente; a
    única opção é reextrair do zero com o material completo), se "origins"
    estiver vazio (v2 corrompido/artesanal sem nenhuma origem) ou se
    new_samples estiver vazio (nada de novo para agregar).
    """
    if "origins" not in imported_profile:
        raise ValueError(
            "Este perfil foi exportado no formato antigo (sem dados por origem) e não "
            "pode ser continuado incrementalmente. Reextraia do zero com o material "
            "completo (original + novo)."
        )
    legacy_origins = imported_profile["origins"]
    if not legacy_origins:
        raise ValueError("Perfil importado não contém nenhuma origem válida (arquivo corrompido).")
    if not new_samples:
        raise ValueError("Nenhuma amostra nova válida para continuar o perfil.")

    legacy_pseudo_samples = [
        {"embedding": o["centroid"], "origin": o["origin"]} for o in legacy_origins
    ]
    new_pseudo_samples = _origin_centroids_as_pseudo_samples(new_samples)
    combined_pseudo_samples = legacy_pseudo_samples + new_pseudo_samples

    centroid = _compute_centroid(combined_pseudo_samples)

    representative = max(new_samples, key=lambda s: s["face"].det_score)
    profile_face = Face(
        bbox=representative["face"].bbox,
        kps=representative["face"].kps,
        det_score=representative["face"].det_score,
    )
    profile_face.embedding = centroid

    new_discarded = new_discarded or []
    total_n_samples = sum(o["n_samples"] for o in legacy_origins) + len(new_samples)

    return {
        "name": name or imported_profile["name"],
        "face": profile_face,
        "thumbnail": representative["thumbnail"],
        "samples": list(new_samples),
        "n_samples": total_n_samples,
        "n_discarded": len(new_discarded),
        "discarded": list(new_discarded),
        "legacy_origins": legacy_origins,
    }


def candidate_as_imported_profile(candidate):
    """Converte uma candidata (retorno de merge_imported_profile, com
    "legacy_origins" + "samples" novos) numa estrutura equivalente a um
    perfil recém-importado (com "origins" combinando ambos) — permite
    encadear merge_imported_profile de novo sobre a candidata (adicionar
    MAIS mídia nova antes de confirmar), em vez de cada chamada partir
    sempre da v_N original e descartar silenciosamente o material de
    rodadas anteriores ainda não confirmadas.

    Usa _combined_origin_summaries (mesma lógica de export_profile) para
    que o resultado tenha exatamente as mesmas origens que um export+import
    real produziria, incluindo a desambiguação de nomes colidentes.
    """
    summaries = _combined_origin_summaries(candidate)
    return {
        "name": candidate["name"],
        "face": candidate["face"],
        "n_samples": candidate["n_samples"],
        "profile_format_version": PROFILE_FORMAT_VERSION,
        "origins": [
            {
                "origin": s["origin"],
                "centroid": s["centroid"],
                "n_samples": s["n_samples"],
                "content_hash": s["content_hash"],
            }
            for s in summaries
        ],
    }


def _combined_origin_summaries(profile, content_hashes=None):
    """Resumos por origem para exportar uma candidata de merge_imported_profile
    (que tem tanto "legacy_origins" quanto "samples" novos) — usada por
    export_profile no lugar de _origin_summaries quando profile tem
    "legacy_origins", para não perder os centroides das origens legadas ao
    reexportar (profile["samples"] só cobre as origens NOVAS).

    Origem nova com o mesmo nome de uma origem legada (ex.: arquivo reenviado
    com o mesmo basename da v_N, mas conteúdo diferente — recorte/reencode,
    que o dedup por hash não pega, ver imported_profile_known_hashes) ganha
    um sufixo numérico para não colidir: duas entradas com o mesmo "origin"
    no .npz fariam essa origem valer o DOBRO do peso normal na próxima
    importação (cada origem pesa 1 na combinação — ver merge_imported_profile).
    """
    legacy_origins = profile.get("legacy_origins", [])
    legacy_summaries = [
        {
            "origin": o["origin"],
            "centroid": o["centroid"],
            "n_samples": o["n_samples"],
            "content_hash": o["content_hash"],
        }
        for o in legacy_origins
    ]
    new_summaries = _origin_summaries(profile.get("samples") or [], content_hashes=content_hashes)

    used_names = {s["origin"] for s in legacy_summaries}
    for summary in new_summaries:
        base_name = summary["origin"]
        if base_name not in used_names:
            used_names.add(base_name)
            continue
        suffix = 2
        candidate_name = f"{base_name}#{suffix}"
        while candidate_name in used_names:
            suffix += 1
            candidate_name = f"{base_name}#{suffix}"
        summary["origin"] = candidate_name
        used_names.add(candidate_name)

    return legacy_summaries + new_summaries
