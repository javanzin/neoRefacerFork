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
VIDEO_FRAME_STRIDE = 15
MAX_FRAMES_PER_VIDEO = 60

# Identifica o espaço vetorial do embedding para validar compatibilidade na
# importação (refacer.py carrega w600k_r50.onnx do pacote buffalo_l). O
# INSwapper só aceita embeddings desse modelo — misturar espaços vetoriais
# produz saída anormal (confirmado na documentação oficial do insightface).
EMBEDDING_MODEL_ID = "buffalo_l/w600k_r50"

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


def _face_sharpness(aligned_crop_bgr):
    gray = cv2.cvtColor(aligned_crop_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


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


def _build_profile_from_samples(samples, name, discarded=None):
    """Núcleo puro (sem estado de instância) de build_profile/build_profiles:
    agrega uma lista de amostras num único perfil. Recebe `samples` e
    `discarded` diretamente em vez de um IdentityProfileBuilder inteiro, para
    não precisar de um objeto parcialmente inicializado (via __new__) só para
    montar o perfil de um cluster já separado.
    """
    if not samples:
        raise ValueError("Nenhuma amostra válida para construir o perfil de identidade.")

    centroid = _compute_centroid(samples)
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

    def add_image(self, frame_bgr, source_label):
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

        if kps is None:
            self.discarded.append({"source": source_label, "reason": "sem landmarks (kps)"})
            return

        if det_score < MIN_DET_SCORE:
            self.discarded.append({
                "source": source_label,
                "reason": f"confiança de detecção baixa ({det_score:.2f})",
            })
            return

        frame_area = frame_bgr.shape[0] * frame_bgr.shape[1]
        bbox_area = max(0.0, (bbox[2] - bbox[0])) * max(0.0, (bbox[3] - bbox[1]))
        face_area_ratio = bbox_area / frame_area if frame_area > 0 else 0.0
        if frame_area <= 0 or face_area_ratio < MIN_FACE_AREA_RATIO_HARD:
            self.discarded.append({"source": source_label, "reason": "rosto pequeno demais no quadro"})
            return

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
            return

        sharpness = _face_sharpness(aligned)
        if sharpness < MIN_SHARPNESS:
            self.discarded.append({
                "source": source_label,
                "reason": f"imagem desfocada (nitidez {sharpness:.0f})",
            })
            return

        if face_area_ratio < MIN_FACE_AREA_RATIO:
            # Faixa intermediária: rosto pequeno só entra com evidência dupla
            # de qualidade (score e nitidez bem acima do mínimo padrão), já
            # que o upscale pro embedding tem menos margem de erro.
            if det_score < MIN_DET_SCORE_COMPENSATED or sharpness < MIN_SHARPNESS_COMPENSATED:
                self.discarded.append({
                    "source": source_label,
                    "reason": "rosto pequeno sem compensação suficiente de nitidez/confiança",
                })
                return

        face = Face(bbox=bbox, kps=kps, det_score=det_score)
        face.embedding = embedding

        self.samples.append({
            "embedding": embedding,
            "face": face,
            "thumbnail": aligned,
            "source": source_label,
        })

    def add_video(self, video_path, source_label):
        """Amostra frames de um vídeo (passo fixo, teto de quadros) e alimenta
        cada um em add_image — mesmo filtro de qualidade das imagens, sem
        decodificar o vídeo inteiro nem mantê-lo todo em memória (diferente de
        analyze_video_in_memory, que existe para o caminho de swap, não de
        extração de identidade).
        """
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            self.discarded.append({"source": source_label, "reason": "vídeo não pôde ser aberto"})
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_index = 0
        frames_used = 0

        with tqdm(total=total_frames, desc=f"Extraindo identidade de {source_label}") as pbar:
            while cap.isOpened() and frames_used < MAX_FRAMES_PER_VIDEO:
                flag, frame = cap.read()
                if not flag:
                    break

                if frame_index % VIDEO_FRAME_STRIDE != 0:
                    frame_index += 1
                    pbar.update()
                    continue

                self.add_image(frame, f"{source_label} (frame {frame_index})")
                frames_used += 1
                frame_index += 1
                pbar.update()

        cap.release()

    def build_profile(self, name="Pessoa 1"):
        return _build_profile_from_samples(self.samples, name, self.discarded)

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

    def build_profiles(self, threshold=CLUSTER_SIMILARITY_THRESHOLD):
        """Separa as amostras em clusters por pessoa e constrói um perfil
        (centroide + Face sintético) por cluster, nomeados "Pessoa 1",
        "Pessoa 2"... na ordem de criação dos clusters.
        """
        if not self.samples:
            raise ValueError("Nenhuma amostra válida para construir perfis de identidade.")

        groups = self.cluster_samples(threshold=threshold)
        profiles = [
            _build_profile_from_samples(group, name=f"Pessoa {i + 1}")
            for i, group in enumerate(groups)
        ]

        # Descartes de qualidade (add_image/add_video) pertencem à extração
        # como um todo, não a um cluster específico — anexados apenas ao
        # primeiro perfil para não duplicar a contagem em todos.
        if profiles:
            profiles[0]["n_discarded"] = len(self.discarded)
            profiles[0]["discarded"] = list(self.discarded)

        return profiles


def merge_profiles(profile_a, profile_b, name=None):
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
    profile_face.embedding = _simple_mean_centroid(combined_samples)

    return {
        "name": name or profile_a["name"],
        "face": profile_face,
        "thumbnail": representative["thumbnail"],
        "samples": combined_samples,
        "n_samples": len(combined_samples),
        "n_discarded": profile_a["n_discarded"] + profile_b["n_discarded"],
        "discarded": profile_a["discarded"] + profile_b["discarded"],
    }


def export_profile(profile, output_path):
    """Grava o perfil em .npz — apenas o centroide final, nunca as amostras individuais."""
    face = profile["face"]
    np.savez(
        output_path,
        embedding=face.embedding.astype(np.float32),
        bbox=np.asarray(face.bbox, dtype=np.float32),
        kps=np.asarray(face.kps, dtype=np.float32) if face.kps is not None else np.zeros((5, 2), dtype=np.float32),
        det_score=np.float32(face.det_score),
        name=profile["name"],
        n_samples=np.int32(profile["n_samples"]),
        embedding_model=EMBEDDING_MODEL_ID,
        created_at=np.int64(int(time.time())),
    )
    return output_path


def import_profile(npz_path):
    """Carrega um perfil exportado, validando a compatibilidade do espaço vetorial.

    Levanta ValueError (não silencioso) se o arquivo não for um perfil válido
    ou tiver sido gerado por um modelo de embedding diferente.
    """
    try:
        data = np.load(npz_path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"Arquivo de perfil inválido ou corrompido: {exc}") from exc

    required_keys = {"embedding", "bbox", "kps", "det_score", "name", "embedding_model"}
    if not required_keys.issubset(data.files):
        raise ValueError("Arquivo não contém os campos esperados de um perfil de identidade.")

    embedding_model = str(data["embedding_model"])
    if embedding_model != EMBEDDING_MODEL_ID:
        raise ValueError(
            f"Perfil incompatível: foi gerado com o modelo de embedding "
            f"'{embedding_model}', mas este app usa '{EMBEDDING_MODEL_ID}'. "
            "Misturar espaços vetoriais diferentes produz swaps incorretos."
        )

    face = Face(
        bbox=data["bbox"],
        kps=data["kps"],
        det_score=float(data["det_score"]),
    )
    face.embedding = data["embedding"].astype(np.float32)

    return {
        "name": str(data["name"]),
        "face": face,
        "n_samples": int(data["n_samples"]),
    }
