"""Perfil de identidade facial reutilizável, extraído de múltiplas imagens.

Reusa o mesmo detector (SCRFD) e o mesmo extrator de embedding (ArcFaceONNX,
buffalo_l/w600k_r50) já carregados por Refacer — nenhum modelo novo é
introduzido. O perfil resultante é um insightface.app.common.Face sintético,
com .embedding igual ao centroide L2-normalizado das amostras válidas, para
ser consumido exatamente como um dest_face extraído de uma única foto (ver
prepare_faces em refacer.py).
"""

import sys
import time

import cv2
import numpy as np
from insightface.app.common import Face
from tqdm import tqdm

# recognition/face_align.py é um módulo local do projeto (não pip-instalado),
# o mesmo usado por refacer.py para o alinhamento/warp do swap em si — reusar
# aqui garante que a textura de pele (ver _extract_skin_texture) é extraída
# com o MESMO template de alinhamento usado no restante do pipeline. Este
# insert é idempotente (sys.path aceita entradas repetidas sem efeito
# colateral) e não depende de refacer.py já ter sido importado antes.
sys.path.insert(1, "./recognition")
import face_align

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
# rostos pequenos "bons" (nítidos, bem detectados) só por serem pequenos.
# Abaixo do piso (MIN_FACE_AREA_RATIO_HARD), o upscale pode gerar
# ringing/artefatos de compressão que o Laplaciano lê como borda real, então o
# corte é mais rígido — mas não absoluto: uma foto de origem excelente
# (score e nitidez bem acima até do patamar "compensado" já exigido na faixa
# intermediária) ainda entra, porque nesse caso a imagem de origem tem
# resolução/qualidade suficiente para o upscale não comprometer o embedding.
# Entre o piso e MIN_FACE_AREA_RATIO, o corte é o padrão "compensado": um
# único sinal de qualidade bem acima do mínimo (score OU nitidez) já basta.
MIN_FACE_AREA_RATIO_HARD = 0.0025  # abaixo disso, só entra com qualidade excepcional
MIN_DET_SCORE_COMPENSATED = 0.75  # exigido p/ rosto pequeno na faixa intermediária
MIN_SHARPNESS_COMPENSATED = 90.0  # idem, ~1.5x o mínimo padrão
MIN_DET_SCORE_EXCEPTIONAL = 0.85  # exigido p/ rosto abaixo do piso absoluto
MIN_SHARPNESS_EXCEPTIONAL = 120.0  # idem, ~2x o mínimo padrão

# face_area_ratio pune injustamente fotos de alta resolução: um rosto de
# 200x200px numa foto 4K ocupa uma fração minúscula do frame mas tem pixels
# reais de sobra para o embedding e pros traços finos de expressão (olheiras,
# sulcos, pés de galinha) sobreviverem ao upscale para 112x112 exigido pelo
# ArcFace. MIN_FACE_SIDE_PX é um piso alternativo em pixels reais do bbox
# (antes do upscale): se o lado mais curto do rosto já tem essa contagem de
# pixels, a exceção "excepcional" acima do piso de área é liberada mesmo sem
# bater score/nitidez elevados — a informação de origem já é suficiente por
# si só, o upscale não está inventando detalhe.
MIN_FACE_SIDE_PX = 90

# Piso permissivo de cada constante acima, usado quando quality_strictness=0
# (slider de rigor todo pra permissivo — ver IdentityProfileBuilder.__init__).
# Escolhidos para ainda rejeitar lixo óbvio (rosto irreconhecível, blur
# extremo) sem impor a régua elevada pensada para o caso comum; a régua real
# em cada extração é uma interpolação linear entre este piso e o valor de
# módulo acima, de acordo com quality_strictness.
MIN_DET_SCORE_FLOOR = 0.2
MIN_SHARPNESS_FLOOR = 15.0
MIN_FACE_AREA_RATIO_FLOOR = 0.003
MIN_FACE_AREA_RATIO_HARD_FLOOR = 0.0005
MIN_DET_SCORE_COMPENSATED_FLOOR = 0.5
MIN_SHARPNESS_COMPENSATED_FLOOR = 60.0
MIN_DET_SCORE_EXCEPTIONAL_FLOOR = 0.5
MIN_SHARPNESS_EXCEPTIONAL_FLOOR = 60.0


def _scaled_threshold(strictness, floor, default):
    """Interpola linearmente entre floor (strictness=0) e default
    (strictness=1) — ver quality_strictness em IdentityProfileBuilder."""
    return floor + (default - floor) * strictness


def _would_pass_at_strictness(metrics, strictness):
    """Reavalia offline se uma amostra descartada por qualidade (ver
    quality_metrics em IdentityProfileBuilder.discarded) passaria com outro
    quality_strictness, sem reprocessar a imagem original — usado para dar
    uma prévia de "quantos descartes o slider resgataria" antes do usuário
    reprocessar de fato (ver preview_quality_strictness/app.py).

    Reimplementa a mesma árvore de decisão de _add_face_candidate a partir
    dos valores brutos já salvos; qualquer mudança de regra lá precisa ser
    espelhada aqui.
    """
    det_score = metrics["det_score"]
    band = metrics.get("band")
    if band is None:
        # Descarte por confiança de detecção antes de qualquer outro cálculo
        # (bbox/nitidez/área nunca chegaram a ser computados).
        return det_score >= _scaled_threshold(strictness, MIN_DET_SCORE_FLOOR, MIN_DET_SCORE)

    sharpness = metrics["sharpness"]
    face_area_ratio = metrics["face_area_ratio"]
    face_side_px = metrics["face_side_px"]

    min_sharpness = _scaled_threshold(strictness, MIN_SHARPNESS_FLOOR, MIN_SHARPNESS)
    min_face_area_ratio = _scaled_threshold(strictness, MIN_FACE_AREA_RATIO_FLOOR, MIN_FACE_AREA_RATIO)
    min_face_area_ratio_hard = _scaled_threshold(strictness, MIN_FACE_AREA_RATIO_HARD_FLOOR, MIN_FACE_AREA_RATIO_HARD)
    min_det_score_compensated = _scaled_threshold(strictness, MIN_DET_SCORE_COMPENSATED_FLOOR, MIN_DET_SCORE_COMPENSATED)
    min_sharpness_compensated = _scaled_threshold(strictness, MIN_SHARPNESS_COMPENSATED_FLOOR, MIN_SHARPNESS_COMPENSATED)
    min_det_score_exceptional = _scaled_threshold(strictness, MIN_DET_SCORE_EXCEPTIONAL_FLOOR, MIN_DET_SCORE_EXCEPTIONAL)
    min_sharpness_exceptional = _scaled_threshold(strictness, MIN_SHARPNESS_EXCEPTIONAL_FLOOR, MIN_SHARPNESS_EXCEPTIONAL)

    if face_area_ratio < min_face_area_ratio_hard:
        has_enough_real_pixels = face_side_px >= MIN_FACE_SIDE_PX
        has_exceptional_quality = det_score >= min_det_score_exceptional and sharpness >= min_sharpness_exceptional
        return has_enough_real_pixels or has_exceptional_quality
    if sharpness < min_sharpness:
        return False
    if face_area_ratio < min_face_area_ratio:
        return not (det_score < min_det_score_compensated and sharpness < min_sharpness_compensated)
    return True

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


def _face_sharpness(aligned_crop_bgr, upscale_factor=1.0):
    """Variância do Laplaciano no crop alinhado (112x112).

    upscale_factor (>=1.0): fator de ampliação aplicado ao crop ORIGINAL do
    bbox para chegar em 112x112 — ver _add_face_candidate. Upscale suaviza
    bordas proporcionalmente ao fator (bordas reais do rosto pequeno ficam
    "esticadas" no resize), então a variância do Laplaciano cai de forma
    artificial mesmo quando a foto de origem é nítida, punindo injustamente
    rostos pequenos. Multiplicar pelo quadrado do fator compensa essa queda,
    aproximando a leitura da nitidez que o crop teria se já nascesse em
    112x112 — sem essa correção, o piso MIN_SHARPNESS_EXCEPTIONAL fica
    praticamente inatingível para qualquer rosto que precise de upscale
    relevante, mesmo com boa qualidade de origem.
    """
    gray = cv2.cvtColor(aligned_crop_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() * (upscale_factor ** 2)


# Template fixo de alinhamento 112x112 do ArcFace (insightface.utils.face_align.
# arcface_dst) — todo "thumbnail" de amostra (ver _add_face_candidate) é
# recortado para este MESMO template, então a posição de olhos/nariz/boca é
# sempre igual entre amostras diferentes: a máscara de pele abaixo pode ser
# fixa, sem depender de detecção adicional por amostra.
_ARCFACE_LANDMARKS_112 = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32,
)

# Separação tom/textura via filtro BILATERAL (não DoG/banda de frequência —
# ver histórico abaixo): o bilateral suaviza preservando bordas de alto
# contraste (pondera vizinhos pela distância espacial E pela diferença de
# valor — um vizinho muito diferente em brilho, como o centro de uma
# olheira, pesa quase nada mesmo estando espacialmente perto), então a
# "textura" (crop menos essa suavização) cai a zero rapidamente ao redor de
# uma mancha isolada, sem os lóbulos de sinal invertido que um filtro linear
# (Gaussian/DoG) produz.
#
# Histórico: a primeira versão usava DoG (blur(sigma=1.0) - blur(sigma=4.0))
# para isolar a banda de frequência de rugas/olheiras (~2-8px no crop
# 112x112). Funcionava para textura DIFUSA (grão fino repetido), mas uma
# olheira/mancha de sombra é uma feature LOCALIZADA de alto contraste — e
# QUALQUER filtro linear (Gaussian, DoG, unsharp mask) produz RINGING ao
# redor desse tipo de feature: um halo de sinal de amplitude menor e SINAL
# INVERTIDO se estendendo pela vizinhança (é uma propriedade matemática da
# subtração de duas convoluções gaussianas, não um parâmetro ajustável).
# Confirmado visualmente numa simulação sintética (olheira + rugas + bigode
# chinês): apareciam anéis concêntricos ao redor de cada olheira que, somados
# ao resultado, coincidiam com a borda da máscara elíptica e liam como "o
# contorno do rosto está aparecendo" — era esse ringing, não vazamento da
# máscara em si.
#
# BILATERAL_SIGMA_COLOR: diferença de brilho (0-255) a partir da qual um
# vizinho passa a pesar pouco na suavização — 25 preserva bem o contraste de
# uma olheira (sombra de ~20-30 níveis mais escura que a pele ao redor) sem
# deixar a suavização "atravessar" a borda da mancha.
# BILATERAL_SIGMA_SPACE: alcance espacial da suavização — 12px cobre a
# escala de uma olheira/banda de sombra (~15-25px de largura no crop 112x112,
# maior que a distância interocular/3) sem borrar rugas finas na direção
# perpendicular a elas.
_SKIN_TEXTURE_BILATERAL_SIGMA_COLOR = 25.0
_SKIN_TEXTURE_BILATERAL_SIGMA_SPACE = 12.0

# Largura (px, no crop 112x112) da rampa suave (smoothstep) nas bordas da
# máscara de pele — tanto ao redor dos círculos de exclusão dos landmarks
# quanto na elipse externa. Uma borda binária (versão anterior) vira um
# degrau visível ("círculo em volta") quando a textura é somada: a transição
# de "textura completa" para "nenhuma textura" acontece em 1px. 4px de rampa:
# - escala para ~10-20px no rosto de destino típico (crop 112 → rosto de
#   200-500px num frame HD), largo o bastante para a transição ficar abaixo
#   do limiar de percepção com amplitudes de textura de ±10 níveis de cinza;
# - estreito o bastante para os círculos de exclusão (raios 11-16px, ver
#   abaixo) mais a rampa não se fundirem no centro do rosto e comerem a
#   bochecha (problema já observado com raio uniforme de 20px);
# - a rampa cresce PARA FORA do raio de exclusão (mask=0 garantido dentro do
#   raio), então não reintroduz cílio/sobrancelha/lábio — só adia o início
#   da textura, em vez de um feather por blur da máscara, que vazaria para
#   dentro da região excluída.
_SKIN_TEXTURE_MASK_FEATHER_PX = 4.0

# Raio (em px, no crop 112x112) de exclusão ao redor de cada landmark, na
# mesma ordem de _ARCFACE_LANDMARKS_112 (olho esq., olho dir., nariz, boca
# esq., boca dir.) — afasta a máscara de pele desses pontos, cuja textura
# própria (cílios, sobrancelha, narinas, contorno labial) não é "pele" e
# desalinharia visivelmente se transplantada para um rosto de destino com
# proporções diferentes. Nariz/boca ficam mais justos para não comer
# bochecha/queixo útil — com um raio único (testado: 20px) os 5 círculos se
# fundem no centro do rosto e a máscara perde toda a região central de
# bochecha. Olhos NÃO usam raio circular — ver _SKIN_TEXTURE_EYE_EXCLUSION
# abaixo, um círculo de 16px aqui cobria também a região de olheira
# (confirmado visualmente: com raio circular a olheira ficava zerada pela
# própria exclusão, antes mesmo de qualquer filtro de extração rodar —
# reportado no teste real como "a olheira não aparece").
_SKIN_TEXTURE_EXCLUSION_RADII = np.array([11.0, 12.0, 12.0], dtype=np.float32)  # nariz, boca esq., boca dir.

# Exclusão dos olhos como ELIPSE achatada verticalmente (não círculo): olho +
# sobrancelha + cílios formam uma faixa HORIZONTAL estreita (mais larga que
# alta), enquanto uma olheira vive numa faixa própria ainda mais abaixo — um
# círculo de raio grande o bastante para cobrir a sobrancelha (que fica ~10px
# ACIMA do centro do olho) automaticamente também cobre a olheira (~7-10px
# ABAIXO), porque o mesmo raio se aplica nas duas direções. A elipse separa
# essas duas distâncias: eixo_x=16 (cobre bem a largura do olho + cantos),
# eixo_y=9 (cobre cílio/sobrancelha imediatamente acima/abaixo, mas termina
# antes da faixa de olheira). offset_y desloca o centro da exclusão ~3px
# para cima do landmark do olho (que marca o centro do OLHO, não da
# sobrancelha) — sem esse deslocamento a mesma elipse simétrica cobriria
# menos sobrancelha do que olheira, quando o objetivo é o oposto.
_SKIN_TEXTURE_EYE_EXCLUSION_AXES = np.array([16.0, 9.0], dtype=np.float32)
_SKIN_TEXTURE_EYE_EXCLUSION_OFFSET_Y = -3.0

# Regiões de INTERESSE (não de exclusão) para _expression_mark_sharpness:
# onde marcas de expressão (olheira, sulco nasolabial/bigode chinês)
# realmente aparecem, para medir nitidez SÓ ali em vez do crop inteiro (ver
# _face_sharpness). Maquiagem/contorno em outra parte do rosto (boca, sombra
# de olho) não deve conseguir inflar essa métrica — restringir a região é o
# que garante isso, ao contrário de tentar detectar maquiagem por cor.
#
# Olheira: mesma faixa ~7-10px ABAIXO do centro do olho já documentada em
# _SKIN_TEXTURE_EYE_EXCLUSION acima (região que a exclusão de olho, com seu
# offset_y=-3, deliberadamente deixa de fora). Elipse achatada na horizontal
# (eixo_x > eixo_y), acompanhando o formato alongado real de uma olheira.
_EXPRESSION_MARK_UNDEREYE_AXES = np.array([12.0, 7.0], dtype=np.float32)
_EXPRESSION_MARK_UNDEREYE_OFFSET_Y = 9.0

# Sulco nasolabial: elipse alongada ligando a lateral do nariz (kps[2], raio
# de exclusão 11px em _SKIN_TEXTURE_EXCLUSION_RADII) ao canto da boca
# correspondente (kps[3]/kps[4], raio 12px) — o próprio sulco vive na faixa
# de pele ENTRE esses dois pontos de exclusão, nunca dentro deles.
# NASOLABIAL_WIDTH: meia-largura da faixa (eixo curto da elipse), estreita o
# bastante para não invadir a bochecha nem o lábio ao lado.
_EXPRESSION_MARK_NASOLABIAL_WIDTH = 6.0


def _ellipse_region_mask(crop_size, center, axes, angle_rad=0.0):
    """Máscara binária (bool, shape (crop_size, crop_size)): True dentro da
    elipse de `center`/`axes` (mesmo grid/fórmula de _skin_region_mask, mas
    como helper genérico reaproveitável — aquela função duplica esta lógica
    inline por landmark em vez de chamar um helper comum).

    angle_rad (opcional): rotaciona a elipse em torno de `center` — necessário
    para uma faixa estreita alinhada a um segmento não-horizontal/vertical
    (ex. sulco nasolabial), onde uma elipse alinhada aos eixos ficaria larga
    demais na direção perpendicular ao segmento real."""
    yy, xx = np.mgrid[0:crop_size, 0:crop_size].astype(np.float32)
    dx, dy = xx - center[0], yy - center[1]
    if angle_rad:
        cos_a, sin_a = np.cos(-angle_rad), np.sin(-angle_rad)
        dx, dy = dx * cos_a - dy * sin_a, dx * sin_a + dy * cos_a
    dist_sq = (dx / axes[0]) ** 2 + (dy / axes[1]) ** 2
    return dist_sq <= 1.0


def _build_expression_marks_mask(crop_size=112):
    """Máscara suave (float 0..1) da UNIÃO das regiões de marca de expressão
    (olheira sob cada olho + sulco nasolabial de cada lado) — usada por
    _expression_mark_sharpness para medir nitidez só onde essas marcas
    aparecem, em vez do crop inteiro.

    Fixa pelo mesmo motivo de _SKIN_REGION_MASK_112: o template de
    alinhamento é fixo, então a posição relativa dessas marcas é sempre a
    mesma entre amostras diferentes.

    LIMITAÇÃO CONHECIDA: rugas de testa não são cobertas — os 5 landmarks do
    ArcFace (_ARCFACE_LANDMARKS_112) não incluem nenhum ponto de testa, então
    essa marca de expressão específica fica fora do escopo geométrico
    possível sem um detector de landmarks mais denso.
    """
    binary_mask = np.zeros((crop_size, crop_size), dtype=bool)

    eye_axes = _EXPRESSION_MARK_UNDEREYE_AXES
    offset_y = _EXPRESSION_MARK_UNDEREYE_OFFSET_Y
    for lx, ly in _ARCFACE_LANDMARKS_112[:2]:
        center = (lx, ly + offset_y)
        binary_mask |= _ellipse_region_mask(crop_size, center, eye_axes)

    nose = _ARCFACE_LANDMARKS_112[2]
    half_width = _EXPRESSION_MARK_NASOLABIAL_WIDTH
    for mouth_corner in _ARCFACE_LANDMARKS_112[3:]:
        delta = mouth_corner - nose
        full_length = float(np.linalg.norm(delta))
        angle = float(np.arctan2(delta[1], delta[0]))
        # Faixa ROTACIONADA para acompanhar o segmento nariz→canto-de-boca
        # (uma elipse alinhada aos eixos, com o mesmo comprimento nas duas
        # direções, cobriria toda a região malar/labial ao redor em vez de só
        # uma faixa estreita ao longo do sulco real — testado e descartado:
        # ruído colocado deliberadamente NA BOCA, fora do sulco, ainda inflava
        # o score). Encolhida ~30% em cada ponta (fator 0.35 de meio-
        # comprimento) para não tocar os próprios landmarks de nariz/boca,
        # que já têm exclusão própria em _SKIN_TEXTURE_EXCLUSION_RADII.
        center = (nose[0] + delta[0] * 0.5, nose[1] + delta[1] * 0.5)
        axes = (full_length * 0.35, half_width)
        binary_mask |= _ellipse_region_mask(crop_size, center, axes, angle_rad=angle)

    sigma = _SKIN_TEXTURE_MASK_FEATHER_PX / 4.0
    mask = cv2.GaussianBlur(binary_mask.astype(np.float32), (0, 0), sigma)
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


_EXPRESSION_MARKS_MASK_112 = _build_expression_marks_mask(112)


def _expression_mark_sharpness(aligned_crop_bgr, upscale_factor=1.0):
    """Variância do Laplaciano RESTRITA às regiões de marca de expressão
    (olheira, sulco nasolabial — ver _build_expression_marks_mask), em vez do
    crop 112x112 inteiro como em _face_sharpness.

    Por quê: nitidez do crop inteiro é um proxy ruim para "marca de expressão
    visível" — qualquer detalhe de alta frequência em OUTRA parte do rosto
    (maquiagem, batom, contorno, sombra) infla o score global do mesmo jeito
    que rugas/olheiras reais inflariam, mesmo que a região da marca em si
    esteja coberta/atenuada. Restringir a métrica à região certa impede que
    detalhe alheio a essas marcas específicas infle o peso da amostra.

    Usada exclusivamente pela ponderação opcional do centroide (ver
    _sharpness_weights/weight_by_sharpness) — os filtros de aceitação de
    amostra (MIN_SHARPNESS*) continuam usando _face_sharpness (crop inteiro),
    que mede um critério diferente: "a foto está em foco o bastante para ser
    usável", não "a marca de expressão está visível".
    """
    gray = cv2.cvtColor(aligned_crop_bgr, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    mask = _EXPRESSION_MARKS_MASK_112
    mean = np.average(laplacian, weights=mask)
    weighted_var = np.average((laplacian - mean) ** 2, weights=mask)
    return float(weighted_var) * (upscale_factor ** 2)


def _smoothstep(t):
    """Rampa suave 0→1 (Hermite, C1-contínua) para t em [0,1]; clampada fora
    do intervalo. Usada nas bordas da máscara de pele — uma rampa linear já
    esconderia o degrau, mas a derivada descontínua nas pontas ainda pode ler
    como "linha" com texturas de amplitude alta; smoothstep custa o mesmo."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _skin_region_mask(crop_size=112):
    """Máscara suave (float 0..1, shape (crop_size, crop_size)) da região de
    pele útil para textura — testa e bochechas do crop alinhado ArcFace,
    excluindo um raio ao redor de cada landmark (olhos, nariz, boca) e a
    borda externa do rosto (fundo/cabelo/orelha, que não é pele facial).

    Gerada como máscara BINÁRIA (elipse + 5 círculos de exclusão) e suavizada
    por um Gaussian blur — mesma técnica já usada nas outras máscaras do
    projeto (ver refacer.py:_rect_cutoff_mask/_mouth_chin_rect_mask, rampa
    smoothstep sobre uma faixa de transição fixa em px) e no blending padrão
    de paste_back de qualquer face-swap. Uma tentativa anterior calculava a
    rampa analiticamente a partir da distância euclidiana até o contorno de
    uma elipse — matematicamente correta em teoria, mas a aproximação de
    gradiente usada para converter "distância normalizada" em "px reais"
    tinha erro sistemático dependente do ângulo (rampa real variando entre
    2.5px e 3.5px em vez dos 4px pretendidos), visível como um "anel" mais
    marcado em certas direções no resultado real. Blur gaussiano sobre uma
    máscara binária não tem esse problema: a largura da rampa em px reais é
    sempre ~3×sigma em qualquer direção, por construção — é uma convolução,
    não uma fórmula geométrica por caso.

    Fixa porque o template de alinhamento é fixo (ver _ARCFACE_LANDMARKS_112)
    — calculada uma vez e cacheada (ver _extract_skin_texture).
    """
    yy, xx = np.mgrid[0:crop_size, 0:crop_size].astype(np.float32)

    # sigma = feather/4: medido empiricamente sobre um degrau 1D (não os
    # "3 sigma" de regra de bolso, que davam ~6.7px de rampa em vez dos 4px
    # pretendidos) — com sigma=1.0 um Gaussian blur produz uma rampa de
    # exatamente 4px entre os pontos em que a máscara cruza 0.01 e 0.99, em
    # qualquer direção (é uma convolução simétrica, não depende da
    # curvatura local do contorno como a tentativa analítica anterior).
    sigma = _SKIN_TEXTURE_MASK_FEATHER_PX / 4.0

    # Sem margem extra no raio de exclusão antes do blur: uma primeira
    # tentativa desta correção usou raio + 1.5*feather para garantir zero
    # estrito bem além do landmark — mas com os raios já grandes (11-16px)
    # isso empurrava as 5 rampas de exclusão a se sobreporem entre si E com
    # a rampa da borda externa da elipse, sufocando quase toda a área de
    # bochecha/testa (cobertura >0.99 caiu de ~25% para ~12%) e criando uma
    # ilha isolada e minúscula de "pele" perto do canto externo do olho, sem
    # conexão com o resto da face — inspecionado visualmente, não só por
    # número. O valor NO PONTO EXATO do landmark já é 0.0 com o raio nominal
    # (confirmado: os raios de 11-16px já são grandes o bastante), então a
    # margem extra não era necessária para a garantia que importa — só
    # empobrecia a cobertura útil.
    center = np.array([56.0, 60.0])
    axes = np.array([46.0, 52.0])
    binary_mask = (
        ((xx - center[0]) / axes[0]) ** 2 + ((yy - center[1]) / axes[1]) ** 2
    ) <= 1.0

    # Olhos: elipse achatada (ver _SKIN_TEXTURE_EYE_EXCLUSION_AXES/_OFFSET_Y)
    # — os 2 primeiros landmarks de _ARCFACE_LANDMARKS_112.
    eye_axes = _SKIN_TEXTURE_EYE_EXCLUSION_AXES
    eye_offset_y = _SKIN_TEXTURE_EYE_EXCLUSION_OFFSET_Y
    for lx, ly in _ARCFACE_LANDMARKS_112[:2]:
        ellipse_dist_sq = ((xx - lx) / eye_axes[0]) ** 2 + ((yy - (ly + eye_offset_y)) / eye_axes[1]) ** 2
        binary_mask &= ellipse_dist_sq > 1.0

    # Nariz e boca (2 cantos): círculo, mesmo tratamento de sempre.
    for (lx, ly), radius in zip(_ARCFACE_LANDMARKS_112[2:], _SKIN_TEXTURE_EXCLUSION_RADII):
        dist_sq = (xx - lx) ** 2 + (yy - ly) ** 2
        binary_mask &= dist_sq > radius ** 2

    mask = cv2.GaussianBlur(binary_mask.astype(np.float32), (0, 0), sigma)

    return np.clip(mask, 0.0, 1.0).astype(np.float32)


_SKIN_REGION_MASK_112 = _skin_region_mask(112)

# Linha de referência (y, no crop 112x112) usada como "pele confirmada" pela
# detecção adaptativa de nascimento de cabelo abaixo — mesma altura usada como
# start da varredura por coluna. 75 fica na bochecha inferior/queixo, região
# que a elipse já cobre com folga em qualquer rosto (bem abaixo dos olhos, em
# y≈51.7) e nunca é tocada por cabelo, só por pele.
_HAIRLINE_SKIN_REFERENCE_Y = 75

# Saltos (em Y/Cr/Cb do YCrCb) a partir dos quais uma coluna deixa de ser
# considerada pele ao subir a partir de _HAIRLINE_SKIN_REFERENCE_Y — ver
# _detect_hairline_top. Qualquer um dos três passar do limiar já conta como
# "achou cabelo" (OR, não AND): cabelo escuro NEUTRO (preto/castanho, o caso
# mais comum) pode ter crominância (Cr/Cb) quase idêntica à de várias peles —
# só muda MUITO em luminância (Y) — enquanto cabelo LOIRO pode ter luminância
# parecida com a pele, mudando sobretudo em crominância. Nenhum canal sozinho
# cobre os dois casos (confirmado empiricamente: Cr sozinho falhava em
# cabelo preto/castanho neutro, testado com pele clara vs. cabelo escuro
# neutro — diferença de Cr de só 19, abaixo de qualquer limiar que não
# dispare em ruído).
#
# Trade-off aceito: Y_THRESHOLD baixo o bastante para pegar cabelo escuro
# sutil também dispara em sombra facial real muito forte (>50% de queda de
# luminância) — nesse caso a máscara perde um pouco de área de testa em troca
# de nunca vazar visivelmente sobre cabelo. Efeito colateral é perda de
# detalhe de textura numa faixa pequena, não uma mancha nova — assimetria de
# risco aceitável frente ao bug relatado (contorno visível sobre cabelo).
_HAIRLINE_Y_JUMP_THRESHOLD = 45.0
_HAIRLINE_CR_JUMP_THRESHOLD = 12.0
_HAIRLINE_CB_JUMP_THRESHOLD = 12.0

# Sigma do blur horizontal aplicado ao perfil de linha-de-cabelo detectado
# por coluna — a linha de cabelo real é contínua (não pula pixel a pixel), e
# sem suavização uma única coluna ruidosa (reflexo, fio solto) criaria um
# entalhe pontudo na máscara.
_HAIRLINE_SMOOTH_SIGMA_PX = 3.0


def _detect_hairline_top(aligned_bgr, center, axes):
    """Para cada coluna dentro do envelope horizontal da elipse de pele,
    varre de baixo (_HAIRLINE_SKIN_REFERENCE_Y, pele confirmada) para cima e
    para no primeiro salto de luminância OU crominância (Y/Cr/Cb do YCrCb,
    ver limiares acima) — ou seja, a borda real de pele-para-cabelo naquela
    coluna, em vez do topo fixo da elipse (que assume uma testa mais
    alta/larga do que a de muitos rostos, deixando a textura vazar sobre
    cabelo — reportado visualmente como um contorno/"triângulo" na testa).

    Nunca sobe além do topo nominal da elipse (`center`/`axes`, mesmo formato
    de _skin_region_mask): a detecção só pode ENCOLHER a região de pele em
    relação à elipse original, nunca alargá-la para fora do envelope já
    validado (evita, por exemplo, vazar para dentro do fundo caso a franja
    seja muito clara e passe despercebida pelo limiar).

    Retorna um array (crop_size,) de floats — y do limite superior de pele
    por coluna, já suavizado horizontalmente. Colunas fora do eixo X da
    elipse mantêm o topo nominal (irrelevantes: a elipse já as exclui).
    """
    ycrcb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    luma, cr, cb = ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]
    h, w = luma.shape
    cx, cy = center
    ax, ay = axes
    nominal_top = cy - ay
    top_limit = np.full(w, nominal_top, dtype=np.float32)

    ref_y = _HAIRLINE_SKIN_REFERENCE_Y
    for x in range(w):
        dx_ratio_sq = ((x - cx) / ax) ** 2
        if dx_ratio_sq >= 1.0:
            continue
        dy = ay * np.sqrt(max(0.0, 1.0 - dx_ratio_sq))
        ellipse_top = int(max(0, cy - dy))

        luma_col, cr_col, cb_col = luma[:, x], cr[:, x], cb[:, x]
        ref_luma, ref_cr, ref_cb = luma_col[ref_y], cr_col[ref_y], cb_col[ref_y]
        y = ref_y
        while y > ellipse_top:
            jumped = (
                abs(float(luma_col[y - 1]) - ref_luma) > _HAIRLINE_Y_JUMP_THRESHOLD
                or abs(float(cr_col[y - 1]) - ref_cr) > _HAIRLINE_CR_JUMP_THRESHOLD
                or abs(float(cb_col[y - 1]) - ref_cb) > _HAIRLINE_CB_JUMP_THRESHOLD
            )
            if jumped:
                break
            y -= 1
        top_limit[x] = y

    return cv2.GaussianBlur(
        top_limit.reshape(1, -1), (0, 0), _HAIRLINE_SMOOTH_SIGMA_PX,
    ).flatten()


def _skin_region_mask_adaptive(aligned_bgr, crop_size=112):
    """Variante de _skin_region_mask que recorta a borda superior da elipse
    pela linha de nascimento de cabelo REAL do rosto em `aligned_bgr` (ver
    _detect_hairline_top), em vez de assumir a mesma testa alta/larga para
    todo mundo. Mantém as mesmas exclusões de olhos/nariz/boca e o mesmo
    feathering (_smoothstep) de _skin_region_mask — só a fronteira externa
    superior passa a variar por rosto.

    Implementada como uma segunda passada de smoothstep sobre a distância
    vertical até a linha detectada (em vez de reconstruir a máscara do zero):
    reaproveita _SKIN_REGION_MASK_112 como teto (nunca alarga a região em
    relação a ela) e só reduz onde a linha detectada estiver ABAIXO do topo
    nominal da elipse naquela coluna.
    """
    center = np.array([56.0, 60.0])
    axes = np.array([46.0, 52.0])
    hairline_top = _detect_hairline_top(aligned_bgr, center, axes)

    yy, xx = np.mgrid[0:crop_size, 0:crop_size].astype(np.float32)
    feather = _SKIN_TEXTURE_MASK_FEATHER_PX
    # Mesma rampa smoothstep de _skin_region_mask (ver _smoothstep), agora
    # medida a partir da linha de cabelo detectada por coluna em vez de uma
    # elipse fixa: 0 em cima da linha, 1 a partir de `feather` px abaixo dela.
    dist_below_hairline = yy - hairline_top[np.newaxis, :]
    hairline_mask = _smoothstep(dist_below_hairline / feather)

    return np.clip(_SKIN_REGION_MASK_112 * hairline_mask, 0.0, 1.0).astype(np.float32)


def _extract_skin_texture(frame_bgr, kps):
    """Extrai a textura de pele (frequência alta — rugas, olheiras, poros) de
    uma foto, para transplante opcional sobre o resultado do swap (ver
    refacer.Refacer._apply_skin_texture).

    IMPORTANTE: recebe o frame ORIGINAL + os 5 landmarks (kps), não o
    "thumbnail" da amostra (sample["thumbnail"], ver _add_face_candidate) —
    esse thumbnail é só um recorte do bbox redimensionado para 112x112, SEM
    alinhamento por landmarks (sem rotação/escala normalizada). A máscara de
    exclusão abaixo (_SKIN_REGION_MASK_112) só faz sentido no espaço do
    template arcface_src fixo (ver _ARCFACE_LANDMARKS_112) — um crop de bbox
    tem os olhos/nariz/boca em posições que variam por pose/enquadramento, o
    que faria os círculos de exclusão sistematicamente errar a posição real
    das feições (apagando olheira, deixando vazar cílio/sobrancelha). Por
    isso este passo refaz o alinhamento aqui via face_align.norm_crop, com o
    MESMO template usado por refacer.py no restante do pipeline de swap.

    LUMINÂNCIA, não BGR: a versão anterior extraía a "alta frequência" por
    canal de cor separado (crop BGR menos seu blur BGR) — isso não separa cor
    de brilho, então variação LOCAL de cor da âncora (sardas avermelhadas,
    ruído de croma do sensor, artefatos de chroma-subsampling do JPEG,
    resquício de maquiagem) vazava para dentro da "textura" e era somada ao
    frame de destino, produzindo mancha de tonalidade visível. Extrair em
    escala de cinza garante que a textura é um delta puro de brilho: ao ser
    somado igualmente aos 3 canais do destino (ver _apply_skin_texture), a
    crominância do destino fica intocada por construção.

    Separação por filtro BILATERAL (não banda de frequência linear — ver
    _SKIN_TEXTURE_BILATERAL_SIGMA_COLOR/_SPACE para o porquê): a "textura"
    é o crop menos sua versão suavizada preservando bordas, o que isola
    rugas/olheiras/poros sem o ringing que um filtro linear produziria ao
    redor de uma mancha localizada de alto contraste (uma olheira).

    MÁSCARA ADAPTATIVA (_skin_region_mask_adaptive): a elipse fixa de
    _SKIN_REGION_MASK_112 assume uma testa mais alta/larga do que a de muitos
    rostos — como o template arcface_src é o mesmo para qualquer pessoa
    (normaliza olhos/nariz/boca, não a linha do cabelo, que não é um
    landmark), a mesma elipse serve de teto para todo mundo mas não se ajusta
    a testa curta/franja baixa, deixando a textura vazar sobre cabelo
    (reportado visualmente como um contorno/"triângulo" na testa). A variante
    adaptativa detecta a linha real de pele-para-cabelo por crominância (ver
    _detect_hairline_top) e só ENCOLHE a elipse onde necessário — nunca a
    alarga além dela.

    Retorna um dict {"texture": array float32 (112,112) — delta de LUMINÂNCIA
    já mascarado pela região de pele, "mask": a máscara adaptativa usada
    (varia por rosto, não é mais sempre _SKIN_REGION_MASK_112)} — "mask"
    viaja junto para quem for aplicar não precisar recalculá-la.
    """
    aligned = face_align.norm_crop(frame_bgr, kps, image_size=112)
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY).astype(np.float32)
    diameter = int(_SKIN_TEXTURE_BILATERAL_SIGMA_SPACE * 3) | 1  # ímpar, exigido pelo bilateralFilter
    smooth = cv2.bilateralFilter(
        gray, diameter, _SKIN_TEXTURE_BILATERAL_SIGMA_COLOR, _SKIN_TEXTURE_BILATERAL_SIGMA_SPACE,
    )
    band = gray - smooth
    mask = _skin_region_mask_adaptive(aligned)
    masked_texture = band * mask
    return {"texture": masked_texture.astype(np.float32), "mask": mask}


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


def _sharpness_weights(samples):
    """Pesos relativos por nitidez LOCALIZADA para a ponderação opcional do
    centroide (weight_by_sharpness): amostras onde a região de marca de
    expressão (olheira, sulco nasolabial — ver _expression_mark_sharpness)
    está mais nítida pesam mais na média; amostras onde essa região específica
    está mole (desfocada, coberta) diluem menos.

    Usa "expression_sharpness" (nitidez restrita a essas regiões), não
    "sharpness" (nitidez do crop inteiro, usada só nos filtros de aceitação de
    amostra) — nitidez do crop inteiro é um proxy ruim para "marca de
    expressão visível": maquiagem/contorno em OUTRA parte do rosto (boca,
    sombra de olho) infla o score global do mesmo jeito que rugas/olheiras
    reais inflariam, mesmo com a marca em si coberta. Restringir à região
    certa evita esse falso-positivo.

    - sqrt comprime a faixa dinâmica: a variância do Laplaciano varia ordens
      de magnitude entre fotos, e usar o valor cru deixaria uma única foto
      ultranítida dominar o centroide (o oposto do objetivo de ter várias
      referências).
    - Amostra sem o campo "expression_sharpness" (legada, anterior a este
      campo, ou pseudo-sample de origem) recebe a mediana das que têm —
      neutra, nunca quebra nem zera ninguém.
    - Sem nenhuma nitidez conhecida, todos os pesos são 1 (idêntico à média
      simples).
    """
    values = np.array([float(s.get("expression_sharpness") or 0.0) for s in samples])
    known = values > 0
    if not np.any(known):
        return np.ones(len(samples))
    values[~known] = np.median(values[known])
    weights = np.sqrt(values)
    return weights / weights.mean()


def _simple_mean_centroid(samples, weights=None):
    """Média de embeddings individualmente L2-normalizados, renormalizada no
    final — usada como ponto de partida do centroide robusto e diretamente
    por quem precisa da decisão "essa amostra é da mesma pessoa?" sem
    suprimir nenhuma amostra (cluster_samples() e merge_profiles()).

    weights (opcional): pesos relativos por amostra (ex. _sharpness_weights).
    None (default) mantém a média simples de sempre, bit-a-bit.
    """
    embeddings = np.stack([s["embedding"] for s in samples])
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = embeddings / norms

    if weights is None:
        centroid = normalized.mean(axis=0)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        centroid = (normalized * weights[:, np.newaxis]).sum(axis=0) / weights.sum()
    centroid_norm = np.linalg.norm(centroid)
    return centroid / centroid_norm if centroid_norm > 0 else centroid


def _compute_centroid(samples, iterations=ROBUST_CENTROID_ITERATIONS, base_weights=None):
    """Centroide robusto: parte da média simples dos embeddings (L2-normalizados
    individualmente) e refina por `iterations` passos, reponderando cada
    amostra por max(0, similaridade_de_cosseno_ao_centroide - piso). Amostras
    mais parecidas com o grupo pesam mais; outliers (ver
    ROBUST_CENTROID_SIMILARITY_FLOOR) pesam ~0 sem ser removidos da lista.

    base_weights (opcional): pesos relativos por amostra (ex.
    _sharpness_weights) multiplicados ao peso de similaridade em cada
    iteração (e usados na média inicial e no atalho de poucas amostras).
    None (default) mantém o comportamento de sempre, bit-a-bit.

    Usada para o perfil final (build_profile/build_profiles) — não para a
    decisão de clustering nem para merge_profiles(), que usam a média simples
    direto (ver _simple_mean_centroid) para não suprimir amostras que já
    foram confirmadas como da mesma pessoa.
    """
    embeddings = np.stack([s["embedding"] for s in samples])
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = embeddings / norms

    if base_weights is not None:
        base_weights = np.asarray(base_weights, dtype=np.float64)

    centroid = _simple_mean_centroid(samples, weights=base_weights)

    if len(samples) <= 3:
        # Com poucas amostras, "outlier" vira decisão por maioria simples
        # (ex.: 2x1) sem base estatística real para distinguir ruído de
        # variação legítima — mantém a média (simples ou ponderada) de sempre.
        return centroid

    for _ in range(iterations):
        similarities = normalized @ centroid
        weights = np.clip(similarities - ROBUST_CENTROID_SIMILARITY_FLOOR, 0.0, None)
        if base_weights is not None:
            weights = weights * base_weights

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


def _origin_centroids_as_pseudo_samples(samples, weight_by_sharpness=False):
    """Um "pseudo-sample" (dict com `embedding` e `origin_weight`) por origem
    distinta, cada um sendo o centroide simples (sem supressão de outlier)
    daquela origem — reaproveita _simple_mean_centroid em vez de duplicar a
    lógica de normalização. Alimentar isso de volta em _compute_centroid faz
    a supressão de outlier operar sobre origens, não sobre frames
    individuais.

    `origin_weight` = sqrt(n_frames_da_origem) (ver _origin_weights): uma
    origem com 1 amostra (foto avulsa) pesa 1; um vídeo com 200 frames pesa
    ~14, não 1 (peso igual = vídeo com várias expressões/poses vale o mesmo
    que uma única foto, subrepresentando a diversidade real que ele capturou)
    nem 200 (frame bruto = vídeo dominando o centroide sobre o conjunto de
    fotos). sqrt comprime a escala pelo mesmo motivo de _sharpness_weights.

    weight_by_sharpness (default False): pondera as amostras DENTRO de cada
    origem pela nitidez (ver _sharpness_weights). Não interfere no
    `origin_weight` — nitidez é relativa dentro da origem, volume é relativo
    entre origens, os dois pesos são ortogonais.
    """
    groups = _group_samples_by_origin(samples)
    return [
        {
            "embedding": _simple_mean_centroid(
                group_samples,
                weights=_sharpness_weights(group_samples) if weight_by_sharpness else None,
            ),
            "origin_weight": np.sqrt(len(group_samples)),
        }
        for group_samples in groups.values()
    ]


def _origin_weights(pseudo_samples):
    """Extrai `origin_weight` dos pseudo-samples de _origin_centroids_as_pseudo_samples
    como array, para uso como base_weights em _compute_centroid."""
    return np.array([s["origin_weight"] for s in pseudo_samples], dtype=np.float64)


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


def _compute_balanced_centroid(samples, weight_by_sharpness=False):
    """Centroide robusto (mesma supressão de outlier de _compute_centroid),
    mas operando sobre um centroide por origem em vez das amostras cruas —
    um vídeo de centenas de frames vira 1 pseudo-sample com peso
    sqrt(n_frames), equalizando sua contribuição à de uma foto avulsa (peso 1,
    por ser sua própria origem com 1 amostra) sem reduzi-la a "vale o mesmo
    que 1 foto" nem deixá-la voltar a dominar por volume bruto — ver
    _origin_centroids_as_pseudo_samples para a motivação do sqrt.

    Com uma única origem, o resultado é a MESMA DIREÇÃO dominante de
    _compute_centroid(samples) (o peso de uma origem única não muda direção,
    só escala, e é normalizado na média), mas não necessariamente bit-a-bit
    idêntico: com 1 pseudo-sample, a função cai no atalho "<=3 amostras" e
    retorna a média simples ponderada direto, sem as iterações de
    reponderação por similaridade que _compute_centroid(samples) aplicaria
    caso houvesse mais de 3 amostras cruas — a única diferença observável é
    de arredondamento de ponto flutuante (ver
    test_single_origin_matches_current_behavior), não de direção.

    weight_by_sharpness: ver _origin_centroids_as_pseudo_samples — pondera
    por nitidez dentro de cada origem, ortogonal ao peso por volume entre
    origens.
    """
    pseudo_samples = _origin_centroids_as_pseudo_samples(samples, weight_by_sharpness=weight_by_sharpness)
    return _compute_centroid(pseudo_samples, base_weights=_origin_weights(pseudo_samples))


def _compute_balanced_mean(samples):
    """Variante de _compute_balanced_centroid SEM supressão de outlier —
    usada por merge_profiles, que deliberadamente evita suprimir qualquer
    amostra (ver docstring de merge_profiles). Ainda assim equaliza a
    contribuição por origem antes da média final, com o mesmo peso
    sqrt(n_frames) por origem (ver _origin_centroids_as_pseudo_samples).
    """
    pseudo_samples = _origin_centroids_as_pseudo_samples(samples)
    return _simple_mean_centroid(pseudo_samples, weights=_origin_weights(pseudo_samples))


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
    weight_by_sharpness=False,
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

    weight_by_sharpness (default False, caminho idêntico ao de antes): pondera
    as amostras pela nitidez LOCALIZADA nas regiões de marca de expressão —
    olheira, sulco nasolabial (ver _sharpness_weights/_expression_mark_sharpness),
    não pela nitidez do crop inteiro — amostras onde essas regiões específicas
    estão mais nítidas pesam mais no centroide. Com balance_by_origin, a
    ponderação acontece dentro de cada origem (não entre origens).
    """
    if not samples:
        raise ValueError("Nenhuma amostra válida para construir o perfil de identidade.")

    if balance_by_origin:
        centroid = _compute_balanced_centroid(samples, weight_by_sharpness=weight_by_sharpness)
    else:
        centroid = _compute_centroid(
            samples,
            base_weights=_sharpness_weights(samples) if weight_by_sharpness else None,
        )
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

    def __init__(self, detector, recognizer, quality_strictness=1.0):
        """Accepts the detector/recognizer directly (SCRFD.detect-compatible
        and ArcFaceONNX.get/compute_sim-compatible objects) rather than a
        live Refacer instance — decouples this module from Refacer's
        internals (see from_refacer() for the app.py call site) and lets
        tests pass simple fakes exposing only .detect/.get/.compute_sim.

        quality_strictness escala os thresholds de qualidade (MIN_DET_SCORE,
        MIN_SHARPNESS etc., ver módulo) linearmente a partir de um piso
        permissivo até o valor de módulo original: 1.0 reproduz exatamente o
        comportamento padrão (mínimo mais rigoroso), 0.0 usa o piso mínimo
        (mais permissivo). Um único fator para todos os critérios em vez de
        expor 8 sliders — a UI expõe só um slider de "rigor" (ver
        extract_identity_profile em app.py).
        """
        self._detector = detector
        self._recognizer = recognizer
        self.quality_strictness = max(0.0, min(1.0, quality_strictness))
        self.samples = []  # list of dict: embedding, face, sharpness, source
        self.discarded = []  # list of dict: source, reason

    @classmethod
    def from_refacer(cls, refacer, quality_strictness=1.0):
        """Convenience constructor for the real app: pulls the already-loaded
        detector/recognizer off a live Refacer instance (refacer.py loads no
        new models for this — see module docstring).
        """
        return cls(refacer.face_detector, refacer.rec_app, quality_strictness=quality_strictness)

    def add_image(self, frame_bgr, source_label, origin=None):
        """origin (opcional): identificador cru da origem (ex. nome do
        arquivo, sem sufixos como "(frame N)") usado para balanceamento por
        origem (ver _build_profile_from_samples/balance_by_origin). Se
        omitido, a própria imagem é sua origem (source_label já é o nome cru
        neste caso — não há sufixo a remover).

        Detecta TODOS os rostos do frame (max_num=0), não só o mais
        proeminente — uma foto de grupo/com mais gente ao fundo tem o rosto
        do alvo descartado por completo se só o mais central/maior fosse
        extraído (o extra nunca chegava nem a virar amostra, silenciosamente
        perdido). Cada rosto vira um candidato próprio, e cluster_samples()
        já existe justamente para separar por pessoa depois — reaproveita a
        mesma extração N-rostos que find_match_in_frame usa para busca
        dirigida (ver docstring lá), só sem o filtro por identidade-alvo.
        """
        if frame_bgr is None:
            self.discarded.append({"source": source_label, "reason": "imagem inválida"})
            return

        bboxes, kpss = self._detector.detect(frame_bgr, max_num=0)
        if bboxes.shape[0] == 0:
            self.discarded.append({"source": source_label, "reason": "nenhum rosto detectado"})
            return

        origin = origin if origin is not None else source_label
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = float(bboxes[i, 4])
            kps = kpss[i] if kpss is not None else None
            label = f"{source_label} (rosto {i + 1})" if bboxes.shape[0] > 1 else source_label
            self._add_face_candidate(frame_bgr, bbox, kps, det_score, label, origin=origin)

    def _add_face_candidate(self, frame_bgr, bbox, kps, det_score, source_label, origin):
        """Núcleo de validação de qualidade + montagem de amostra
        compartilhado entre add_image (todos os rostos do frame, sem filtro
        de identidade) e find_match_in_frame (todos os rostos do frame, só os
        candidatos que baterem com um perfil-alvo).

        origin identifica a origem "crua" da amostra (nome do arquivo/vídeo,
        sem os sufixos de exibição que source_label pode carregar como
        "(frame N)"/"(rosto N)") — usado só para balanceamento por origem,
        nunca para exibição.
        """
        if kps is None:
            self.discarded.append({"source": source_label, "reason": "sem landmarks (kps)"})
            return None

        strictness = self.quality_strictness
        min_det_score = _scaled_threshold(strictness, MIN_DET_SCORE_FLOOR, MIN_DET_SCORE)
        min_sharpness = _scaled_threshold(strictness, MIN_SHARPNESS_FLOOR, MIN_SHARPNESS)
        min_face_area_ratio = _scaled_threshold(strictness, MIN_FACE_AREA_RATIO_FLOOR, MIN_FACE_AREA_RATIO)
        min_face_area_ratio_hard = _scaled_threshold(strictness, MIN_FACE_AREA_RATIO_HARD_FLOOR, MIN_FACE_AREA_RATIO_HARD)
        min_det_score_compensated = _scaled_threshold(strictness, MIN_DET_SCORE_COMPENSATED_FLOOR, MIN_DET_SCORE_COMPENSATED)
        min_sharpness_compensated = _scaled_threshold(strictness, MIN_SHARPNESS_COMPENSATED_FLOOR, MIN_SHARPNESS_COMPENSATED)
        min_det_score_exceptional = _scaled_threshold(strictness, MIN_DET_SCORE_EXCEPTIONAL_FLOOR, MIN_DET_SCORE_EXCEPTIONAL)
        min_sharpness_exceptional = _scaled_threshold(strictness, MIN_SHARPNESS_EXCEPTIONAL_FLOOR, MIN_SHARPNESS_EXCEPTIONAL)

        if det_score < min_det_score:
            self.discarded.append({
                "source": source_label,
                "reason": f"confiança de detecção baixa ({det_score:.2f})",
                "quality_metrics": {"det_score": det_score},
            })
            return None

        frame_area = frame_bgr.shape[0] * frame_bgr.shape[1]
        bbox_area = max(0.0, (bbox[2] - bbox[0])) * max(0.0, (bbox[3] - bbox[1]))
        face_area_ratio = bbox_area / frame_area if frame_area > 0 else 0.0
        if frame_area <= 0:
            self.discarded.append({"source": source_label, "reason": "rosto pequeno demais no quadro"})
            return None

        embedding = self._recognizer.get(frame_bgr, kps)

        face_side_px = min(bbox[2] - bbox[0], bbox[3] - bbox[1])
        crop_bgr = frame_bgr[max(0, int(bbox[1])):int(bbox[3]), max(0, int(bbox[0])):int(bbox[2])]
        aligned = cv2.resize(crop_bgr, (112, 112)) if bbox[3] > bbox[1] and bbox[2] > bbox[0] else None

        if aligned is None:
            # Degenerate bbox — no crop to judge sharpness on or show as a
            # thumbnail. Discard outright instead of falling back to a
            # sharpness value that would always pass the check below.
            self.discarded.append({"source": source_label, "reason": "bbox inválida (sem crop)"})
            return None

        # upscale_factor > 1 só quando o crop original é menor que 112px (o
        # caso comum de rosto pequeno) — crops já maiores que 112px são
        # DOWNSCALED por cv2.resize, o que não sofre da mesma suavização
        # artificial, então o fator fica travado em 1.0 (sem correção) nesse
        # caso, evitando inflar a nitidez de fotos que já eram grandes.
        upscale_factor = max(1.0, 112.0 / face_side_px) if face_side_px > 0 else 1.0
        sharpness = _face_sharpness(aligned, upscale_factor=upscale_factor)
        expression_sharpness = _expression_mark_sharpness(aligned, upscale_factor=upscale_factor)

        if face_area_ratio < min_face_area_ratio_hard:
            # Abaixo do piso absoluto, entra por qualquer uma de duas portas:
            # (a) pixels reais de sobra no bbox original (face_side_px), sinal
            # direto de que o upscale não está inventando detalhe — não
            # depende de score/nitidez, já que a informação de origem já
            # basta por si só; ou (b) qualidade excepcional (score e nitidez
            # bem acima do padrão), para o caso de rosto realmente pequeno em
            # pixels mas ainda assim bem capturado. face_area_ratio pune
            # injustamente fotos de alta resolução (ver MIN_FACE_SIDE_PX) —
            # por isso a porta (a) existe.
            has_enough_real_pixels = face_side_px >= MIN_FACE_SIDE_PX
            has_exceptional_quality = det_score >= min_det_score_exceptional and sharpness >= min_sharpness_exceptional
            if not has_enough_real_pixels and not has_exceptional_quality:
                self.discarded.append({
                    "source": source_label,
                    "reason": (
                        f"rosto pequeno demais no quadro ({face_area_ratio * 100:.2f}% da área, "
                        f"{face_side_px:.0f}px/{MIN_FACE_SIDE_PX}px exigidos; "
                        f"score {det_score:.2f}/{min_det_score_exceptional:.2f}, "
                        f"nitidez {sharpness:.0f}/{min_sharpness_exceptional:.0f} exigidos p/ exceção)"
                    ),
                    "quality_metrics": {
                        "det_score": det_score, "sharpness": sharpness,
                        "face_area_ratio": face_area_ratio, "face_side_px": face_side_px,
                        "band": "hard",
                    },
                })
                return None
        elif sharpness < min_sharpness:
            self.discarded.append({
                "source": source_label,
                "reason": f"imagem desfocada (nitidez {sharpness:.0f})",
                "quality_metrics": {
                    "det_score": det_score, "sharpness": sharpness,
                    "face_area_ratio": face_area_ratio, "face_side_px": face_side_px,
                    "band": "normal",
                },
            })
            return None
        elif face_area_ratio < min_face_area_ratio:
            # Faixa intermediária: rosto pequeno entra com UM sinal de
            # qualidade bem acima do mínimo padrão (score OU nitidez) — exigir
            # os dois simultaneamente descartava fotos de retrato comuns
            # (nítidas, mas com score de detecção só "normal", ou bem
            # detectadas mas com nitidez só "normal") que já eram boas o
            # bastante, sem ganho real de robustez do embedding.
            if det_score < min_det_score_compensated and sharpness < min_sharpness_compensated:
                self.discarded.append({
                    "source": source_label,
                    "reason": "rosto pequeno sem compensação suficiente de nitidez/confiança",
                    "quality_metrics": {
                        "det_score": det_score, "sharpness": sharpness,
                        "face_area_ratio": face_area_ratio, "face_side_px": face_side_px,
                        "band": "intermediate",
                    },
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
            # Nitidez do crop (variância do Laplaciano, a mesma já usada no
            # filtro acima) — usada só nos filtros de aceitação (MIN_SHARPNESS*)
            # e em _would_pass_at_strictness, não na ponderação por nitidez.
            "sharpness": sharpness,
            # Nitidez restrita às regiões de marca de expressão (olheira,
            # sulco nasolabial) — usada exclusivamente pela ponderação
            # opcional do centroide (ver _sharpness_weights,
            # weight_by_sharpness). Campo separado de "sharpness" porque mede
            # um critério diferente: "a marca de expressão está visível
            # aqui", não "a foto está em foco o bastante para ser usável".
            "expression_sharpness": expression_sharpness,
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

    def build_profile(self, name="Pessoa 1", balance_by_origin=False, anchor_sample=None, anchor_weight=ANCHOR_MAX_WEIGHT, weight_by_sharpness=False):
        return _build_profile_from_samples(
            self.samples, name, self.discarded,
            balance_by_origin=balance_by_origin, anchor_sample=anchor_sample, anchor_weight=anchor_weight,
            weight_by_sharpness=weight_by_sharpness,
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

    def preview_quality_strictness(self, strictness):
        """Estima, sem reprocessar nenhuma imagem, quantos dos descartes
        atuais por qualidade teriam passado com outro quality_strictness —
        usado pela UI para orientar o usuário antes de rodar a extração de
        novo com o slider em outra posição (ver identity_quality_strictness
        em app.py). Só considera descartes com quality_metrics (os por
        confiança/nitidez/área); descartes de outra natureza (arquivo
        inválido, nenhum rosto detectado, duplicata) nunca são resgatáveis
        por rigor e ficam de fora da contagem.
        """
        eligible = [d for d in self.discarded if "quality_metrics" in d]
        rescued = sum(1 for d in eligible if _would_pass_at_strictness(d["quality_metrics"], strictness))
        return {"eligible": len(eligible), "rescued": rescued}

    def build_profiles(self, threshold=CLUSTER_SIMILARITY_THRESHOLD, balance_by_origin=False, weight_by_sharpness=False):
        """Separa as amostras em clusters por pessoa e constrói um perfil
        (centroide + Face sintético) por cluster, nomeados "Pessoa 1",
        "Pessoa 2"... na ordem de criação dos clusters.

        balance_by_origin e weight_by_sharpness (default False): repassados a
        _build_profile_from_samples — ver docstring lá para o que muda. Âncora
        não se aplica aqui: ela é escolhida pelo usuário depois de já ver os
        perfis extraídos (ver apply_anchor_to_profile). A ponderação por
        nitidez também não afeta o clustering (cluster_samples segue média
        simples: a pergunta lá é pertencimento, não representatividade).
        """
        if not self.samples:
            raise ValueError("Nenhuma amostra válida para construir perfis de identidade.")

        groups = self.cluster_samples(threshold=threshold)
        profiles = [
            _build_profile_from_samples(
                group, name=f"Pessoa {i + 1}",
                balance_by_origin=balance_by_origin, weight_by_sharpness=weight_by_sharpness,
            )
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

    Retorna a amostra aceita (dict com "embedding" e "skin_texture" — ver
    _extract_skin_texture, compatível com anchor_sample de
    apply_anchor_to_profile) ou None se rejeitada, e a lista de descartes
    (mesmo formato de IdentityProfileBuilder.discarded) para o chamador
    reportar o motivo ao usuário.
    """
    builder = IdentityProfileBuilder(detector, recognizer)
    builder.add_image(frame_bgr, source_label)
    anchor_sample = builder.samples[0] if builder.samples else None
    if anchor_sample is not None:
        # frame_bgr (o quadro ORIGINAL, não o thumbnail de bbox da amostra) +
        # os kps detectados — _extract_skin_texture faz o próprio alinhamento
        # por landmark a partir daqui (ver docstring lá para o porquê).
        anchor_sample["skin_texture"] = _extract_skin_texture(frame_bgr, anchor_sample["face"].kps)
    return anchor_sample, builder.discarded


def apply_anchor_to_profile(profile, anchor_sample=None, anchor_weight=ANCHOR_MAX_WEIGHT, balance_by_origin=False, weight_by_sharpness=False):
    """Reaplica (ou remove, com anchor_sample=None) a foto âncora sobre um
    perfil já construído, recalculando a partir de profile["samples"] em vez
    de acumular — chamar duas vezes seguidas com âncoras diferentes não
    duplica amostras nem cresce n_samples.

    balance_by_origin e weight_by_sharpness devem refletir as mesmas opções
    usadas para construir o perfil originalmente (ver
    _build_profile_from_samples), para que reaplicar a âncora não mude
    silenciosamente essas outras escolhas.

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
        weight_by_sharpness=weight_by_sharpness,
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

    # Textura de pele transplantada (opt-in, ver apply_identity_anchor em
    # app.py) — arrays numéricos puros (float32/mask), nunca pickle, mesma
    # justificativa de segurança de origin_names abaixo. Sem isso, exportar o
    # .npz depois de ligar o transplante perderia o efeito silenciosamente
    # (mesmo bug que a âncora tinha antes de profile["anchor_sample"] existir).
    skin_texture = getattr(face, "skin_texture", None)
    if skin_texture is not None:
        fields.update(
            skin_texture=skin_texture["texture"].astype(np.float32),
            skin_texture_mask=skin_texture["mask"].astype(np.float32),
            skin_texture_intensity=np.float32(skin_texture.get("intensity", 1.0)),
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

        skin_texture_keys = {"skin_texture", "skin_texture_mask", "skin_texture_intensity"}
        if skin_texture_keys.issubset(data.files):
            face.skin_texture = {
                "texture": data["skin_texture"].astype(np.float32),
                "mask": data["skin_texture_mask"].astype(np.float32),
                "intensity": float(data["skin_texture_intensity"]),
            }

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
      tratado como uma etapa separada — não coberto por esta função. Pela
      mesma razão, uma textura de pele transplantada (ver
      app.apply_identity_anchor/_reattach_anchor_extras) TAMBÉM não
      sobrevive a este fluxo — precisa ser reaplicada manualmente depois,
      sobre a candidata resultante, antes de reexportar.

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
