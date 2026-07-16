# Review Geral — Aba de Vídeo (Performance, Cache, Swap e Qualidade)

Escopo: tudo que é usado pela aba de vídeo — `refacer.py` (pipeline completo: análise, cache, swap, encode) e `app.py` (orquestração/Gradio). Este documento **consolida e substitui** `REVIEW_PERFORMANCE_VIDEO.md` como referência única: reverifica cada achado antigo contra o código atual (mantendo os que ainda são reais, removendo os já corrigidos) e adiciona achados novos sobre a lógica de **swap/matching** que não tinham sido cobertos antes.

Apenas mapeamento — nenhuma alteração de código foi aplicada. Detalhe suficiente em cada ponto (arquivo:linha, problema, impacto, sugestão) para correção posterior.

Contexto de hardware assumido: GPU Tesla T4 (16GB VRAM), 15GB RAM, **CPU muito limitada** (Colab/Lightning AI free tier). GPU e RAM são relativamente folgados; CPU é o gargalo real. Achados de qualidade só entram se o ganho compensar um custo de performance baixo/desprezível nesse hardware.

## O que já foi corrigido desde o review anterior (não repetido abaixo)

Confirmado lendo o código atual — mantido aqui só como registro, sem exigir ação:

- **Threads no caminho não-cacheado (antigo #1)**: `reface_group` (`refacer.py:1183-1205`) agora serializa no modo CUDA e só usa `ThreadPoolExecutor` em CPU/CoreML/TensorRT, onde overlap de I/O ainda ajuda.
- **Profiling sempre ligado (antigo #4)**: `profiling_enabled` agora depende de `REFACER_PROFILE` env var (`refacer.py:76`), desligado por padrão.
- **Embedding recomputado por job (antigo #7)**: `analyze_video_in_memory` agora computa embeddings uma única vez junto com bboxes/kpss (`refacer.py:330-340`) e `reface_with_precomputed` os reutiliza (`refacer.py:379-390`) — elimina N-1 passadas de inferência de embedding no fluxo multi-rosto.
- **Encode duplo mp4v→H264 (antigo #8/#12)**: `__convert_video` agora sempre reencoda via H.264/nvenc, com ou sem áudio (`refacer.py:1401-1418`) — não há mais vídeo saindo em mp4v puro.
- **RAM sem teto em `analyze_video_in_memory` (antigo #9)**: agora há `in_memory_analysis_max_mb` (`refacer.py:94-100`) com checagem antes (`refacer.py:300-306`) e durante a decodificação (`refacer.py:318-323`), caindo para o caminho per-job se o vídeo for grande demais.
- **Transição de blending linear (antigo #11)**: `_partial_face_blend` já usa smoothstep (`refacer.py:610-617`).

## Prioridade sugerida (todos os achados abaixo)

0. **#13 / #14** (Parte 4) — validar de verdade o pipe ffmpeg antes de confiar nele: o fallback atual não dispara nos modos de falha reais, e dimensões ímpares quebram o pipe. Resolver **antes** do teste em produção da 2ª tentativa de encode (#8), para não confundir essas falhas com a incompatibilidade da 1ª tentativa.
1. **#6** — logar qual job falhou e não deixar jobs seguintes serem pulados silenciosamente em `app.py` — baixo custo. Impacto real é menor do que parece: jobs *anteriores* ao que falha já ficam no histórico normalmente (o `video_history.append` roda dentro do loop, antes do próximo job); só os jobs *depois* do que falhar são perdidos, sem indicar qual falhou.
2. **#1** — assignment por similaridade global em "Faces By Match" em vez de greedy por ordem de configuração — maior efeito na qualidade percebida do swap com múltiplos rostos-alvo (relevante só se o usuário configura 2+ rostos-alvo simultaneamente nesse modo).
4. **#9 / #10** — I/O pulverizado do `.npz` e hash/capture recomputado no light cache — ganho de CPU real se esses caminhos ainda forem usados na prática (confirmar antes).
5. **#2 / #3** — continuidade de identidade entre frames — maior esforço, avaliar com o usuário antes de implementar.
6. **#7** — `_calculate_optimal_batch_size` não faz o que os prints sugerem — simplificar ou remover a falsa sensação de otimização.
7. **#5** — limitação conhecida do "Reface Ratio" com pose de cabeça — documentar, não implementar por ora. **#12 já implementado** (máscara oval boca→queixo, opt-in).
8. **#19 (novo)** — integridade do swap em pose lateral/deitada — limitação estrutural do alinhamento (`SimilarityTransform` + `INSwapper` de terceiros), custo de mitigar é baixo mas o ganho é só parcial (pose moderada) e não resolve pose extrema — documentar, não implementar por ora.

**Removido desta lista**: o achado antigo sobre `yield` em `app.py:run()` estava incorreto — o `yield` (linha 339) já está dentro do loop `for job in jobs`, então o player já atualiza a cada job processado, como confirmado pelo usuário. Não é um bug.

---

## Parte 1 — Lógica de Swap e Matching (novo, não coberto no review anterior)

### 1. Em "Faces By Match" com múltiplos rostos-alvo, a ordem de `replacement_faces` decide o empate, não a similaridade

- **Arquivo:linha**: `refacer.py:973-986` (`_apply_swaps`, ramo `else`)
- **Problema**: o loop externo itera `rep_face` na ordem de `self.replacement_faces` (ordem das abas "Face #1", "Face #2"... em `app.py`). Para cada `rep_face`, percorre `faces` de trás para frente e troca o **primeiro** rosto detectado cujo `sim >= threshold`, removendo-o da lista (`del faces[i]`) antes de passar para o próximo `rep_face`.
- **Impacto**: se o rosto-alvo #1 e o rosto-alvo #2 têm ambos similaridade acima do threshold contra o mesmo rosto detectado no frame (pessoas parecidas, ou thresholds soltos), o rosto #1 "ganha" esse rosto detectado sempre, mesmo que o #2 tivesse similaridade mais alta. Não há comparação global — é greedy por ordem de configuração, não por melhor match.
- **Cenário concreto de falha**: vídeo com duas pessoas parecidas, usuário configura Face #1 → pessoa A e Face #2 → pessoa B com thresholds baixos (0.2, valor padrão do slider). Em frames onde o embedding erra por ângulo/luz, a pessoa B pode ser trocada pelo destino configurado para a Face #1, gerando swap trocado de forma inconsistente entre frames — "flicker" de identidade no vídeo final.
- **Sugestão**: para N>1 rostos-alvo em "Faces By Match", calcular a matriz de similaridade completa (todos os `rep_face` × todos os `faces` do frame) e resolver como assignment (maior similaridade global, ou um único passe ordenado por `sim` descendente) em vez de greedy por ordem de configuração. Custo desprezível (matriz pequena, poucos rostos por frame).

### 2. Nenhum rastro de identidade entre frames — o "flicker" de swap é estrutural, não só do achado #1

- **Arquivo:linha**: `refacer.py:933-986` (`_apply_swaps`), sem qualquer estado entre chamadas
- **Problema**: cada frame é resolvido do zero, sem tracking de qual rosto detectado no frame N corresponde ao do frame N-1. Em "Faces By Match", se o embedding de um frame específico cair abaixo do threshold (motion blur, rotação momentânea, oclusão parcial), aquele frame não troca aquele rosto — aparece "original" por 1-2 frames no meio de um trecho trocado, artefato bem visível a 24-30fps.
- **Impacto**: mais perceptível em vídeos com movimento de câmera/cabeça e em "Faces By Match" com thresholds conservadores (o usuário tende a subir o threshold para evitar trocas erradas do achado #1, o que agrava este problema por outro lado).
- **Sugestão** (custo mais alto — registrar, não implementar sem validar com o usuário): manter o último `rep_face` casado por posição/IoU de bbox entre frames consecutivos como fallback quando a similaridade cai abaixo do threshold mas a posição do rosto é consistente com o frame anterior. Evita o "pisca" sem precisar de um tracker completo (SORT/DeepSORT é overkill para este hardware). Precisa de estado por-vídeo corretamente resetado entre runs.

### 3. "Multiple Faces" e o fallback por posição dependem inteiramente de `bbox[0]` — sensível a movimento horizontal

- **Arquivo:linha**: `refacer.py:942` (`faces = sorted(faces, key=lambda face: face.bbox[0])`), usado pelos ramos `multiple_faces_mode` (`944-953`) e `disable_similarity` com múltiplos destinos (`954-964`)
- **Problema**: a correspondência rosto-detectado ↔ rosto-de-destino é feita puramente por ordem horizontal (esquerda→direita) no frame atual, sem continuidade. Se duas pessoas se cruzam (uma passa na frente/atrás da outra, ou trocam de ordem relativa ao se mover), os destinos "trocam de dono" instantaneamente no frame em que a ordem de `bbox[0]` se inverte.
- **Impacto**: em "Multiple Faces" (modo que explicitamente não usa similaridade, então não tem outra forma de decidir), isso é uma troca visível de rosto sempre que duas pessoas se cruzam — não é caso raro em vídeos com mais de uma pessoa se movendo.
- **Sugestão**: mesmo raciocínio do achado #2 — fallback de continuidade por proximidade ao bbox do frame anterior (em vez de reordenar do zero a cada frame) resolve o cruzamento sem precisar de embedding/similaridade (que "Multiple Faces" evita por design). Confirmar com o usuário se esse comportamento já foi notado como problema antes de investir.

### 4. ~~`run()` só entrega o resultado do último job~~ — **descartado, não é um bug**

Achado removido após verificação: o `yield` (`app.py:339`) já está dentro do `for job in jobs` (linhas 316-339), não depois dele. O player já é atualizado incrementalmente a cada job processado, confirmado em uso real pelo usuário. Mantido aqui só para registro de que foi investigado e descartado.

### 5. `partial_reface_ratio` (corte parcial de rosto) usa a bbox do detector para todos os modos — geometria de corte não vale quando o rosto está de perfil

- **Arquivo:linha**: `refacer.py:582-625` (`_partial_face_blend`), `cutoff = int(h * (1.0 - self.blend_height_ratio))` na linha 597
- **Problema**: o corte é sempre uma linha horizontal a uma fração fixa da altura da bbox, independente da orientação da cabeça. Em rosto de perfil/inclinado, a bbox já não é um quadrado alinhado ao rosto — a "metade inferior" da bbox pode não corresponder à metade inferior real do rosto (queixo/boca), tornando o corte inconsistente conforme a cabeça se move.
- **Impacto**: afeta só quem usa o slider "Reface Ratio" > 0 (recurso opt-in). Quando usado, a consistência do corte varia com a pose da cabeça — mais notado em vídeos com giro de cabeça, onde a linha de corte parece "flutuar" em relação a boca/queixo entre frames.
- **Sugestão**: mesmo racional já discutido para a máscara elíptica (ver achado #11 abaixo) — resolver corretamente exigiria estimar pose (yaw/pitch a partir dos 5 keypoints) e rotacionar a linha de corte, o que é mais trabalho do que esse recurso opt-in/uso ocasional justifica agora. Registrar como limitação conhecida.

### 6. `prepare_faces` lança exceção dura se qualquer imagem de destino não tiver rosto detectável — interrompe todos os jobs já enfileirados

- **Arquivo:linha**: `refacer.py:1115-1145` (`prepare_faces`), `raise Exception('No face detected on "Destination face" image')` na linha 1127; chamado a cada job em `reface`/`reface_with_precomputed`/`reface_with_light_cache`/`reface_with_cache`
- **Problema**: em "Faces By Match" com múltiplos jobs (`_build_video_face_jobs` monta 1 job por combinação origem×destino), se o destino de um job específico não tiver rosto detectável (imagem cortada, ângulo ruim, arquivo corrompido), a exceção interrompe o `for job in jobs` inteiro em `app.py:316-336` — jobs anteriores ao que falhou já produziram vídeo e foram gravados no histórico, mas os jobs restantes nunca rodam, sem mensagem clara sobre qual job falhou.
- **Impacto**: com N rostos-alvo configurados, um destino problemático em qualquer posição da lista aborta silenciosamente o processamento dos rostos-alvo seguintes.
- **Sugestão**: no loop de `app.py:316`, envolver a chamada a `refacer.reface(...)` num try/except por job, logar/expor qual `job['label']` falhou e continuar para o próximo job em vez de abortar o lote inteiro. Custo baixo, ganho de robustez alto.

---

## Parte 2 — Performance, Cache e Encode (reverificado do review anterior)

### 7. `_calculate_optimal_batch_size` não controla batch de inferência — só o tamanho do flush do writer

- **Arquivo:linha**: `refacer.py:542-580`, usada em `refacer.py:375, 457, 1039, 1342`
- **Status**: ainda presente, não corrigido.
- **Problema**: a função calcula `batch_size` a partir de VRAM/FPS, mas a inferência é frame a frame — `batch_size` só controla de quantos em quantos frames o código acumula em RAM antes de um `output.write` em lote. Não há batch de inferência real, e o `VideoWriter` já bufferiza sozinho.
- **Impacto**: os prints (`[BATCH] Dynamic batch size...VRAM`) sugerem otimização de GPU que não existe. Em vídeos maiores, segurar até 1000 frames Full HD em RAM (~6GB) antes de escrever pressiona os 15GB sem ganho de velocidade.
- **Sugestão**: para ganho real de GPU seria preciso batching de inferência de verdade (empilhar N crops num único tensor para `inswapper`/`arcface`). Caso não valha o esforço agora, simplificar: escrever cada frame direto no writer e remover a lista intermediária, eliminando custo de RAM e a falsa sensação de otimização.

### 8. Encode intermediário em `mp4v` — histórico da tentativa anterior + **segunda tentativa em andamento (não validada em produção)**

- **Arquivo:linha (estado original antes desta tentativa)**: escrita mp4v em `refacer.py:372, 453, 1029, 1338`; `__convert_video` em `refacer.py:1401-1418`
- **Histórico**: o usuário já tentou uma vez escrever direto em H.264 (trocando o codec do `cv2.VideoWriter`) e reverteu porque uma lib do projeto (não identificada com precisão — "lib antiga", só no vídeo) era incompatível com esse encoding direto. O mp4v existia propositalmente como intermediário "seguro" antes do reencode em `__convert_video`, não por descuido.
- **O que foi tentado agora (2ª tentativa, mecanismo diferente da 1ª)**: em vez de pedir ao `cv2.VideoWriter` para gravar H.264 diretamente (via `fourcc`), foi implementado um pipe de frames crus (`rawvideo`/`bgr24`) direto para um processo `ffmpeg` externo via `subprocess.Popen`, que faz o único encode final em H.264/nvenc (ou o encoder que `__check_encoders` já detectou). Essa via não depende do build do OpenCV ter suporte a H.264 — usa o mesmo binário `ffmpeg` que o projeto já usa em `__convert_video`/`__check_encoders`, então tende a não bater na mesma limitação da 1ª tentativa (mas isso **ainda não foi confirmado** em teste real).

**Mudanças de código feitas (todas em `refacer.py`, nenhuma revertida ainda):**

1. Nova classe `_FfmpegVideoWriter` (perto do topo do arquivo, após `class RefacerMode`): abre um `subprocess.Popen(['ffmpeg', ...])` com stdin em pipe, escreve frames via `.write(frame)` chamando `frame.tobytes()`, e finaliza via `.release()` fechando o stdin e esperando o processo (`.wait()`). Usa uma thread daemon (`_drain_stderr`) para drenar o stderr do ffmpeg continuamente num `deque(maxlen=200)`, evitando deadlock por buffer de pipe cheio; se o `write()` ou o `release()` falharem (`BrokenPipeError` ou `returncode != 0`), levanta `RuntimeError` com a cauda do stderr do ffmpeg anexada.
2. Novo método `_open_video_writer(self, output_path, fps, frame_width, frame_height)`: tenta abrir o `_FfmpegVideoWriter` primeiro; se a construção falhar (`except Exception`), cai para o `cv2.VideoWriter` + mp4v antigo. Retorna `(writer, already_encoded)` — `already_encoded=True` quando o pipe direto foi usado.
3. Os 4 pontos que antes criavam `cv2.VideoWriter` com `fourcc='mp4v'` diretamente (`reface_with_precomputed`, `reface_with_light_cache`, `reface_with_cache`, e o caminho sem cache em `reface`) foram trocados para chamar `_open_video_writer(...)` e propagar `already_encoded` até a chamada de `__convert_video`.
4. `__convert_video` ganhou um parâmetro novo `already_encoded=False`. Quando `True`: se não há áudio no vídeo original, o arquivo já está pronto e é devolvido sem nenhum passo de ffmpeg adicional; se há áudio, faz *apenas* a mux do áudio original com `-c:v copy` (sem reencodar o vídeo) em vez do reencode completo `vcodec=self.ffmpeg_video_encoder` que rodava antes.
5. Imports adicionados no topo do arquivo: `deque` (de `collections`) e `threading`.

**Ainda não testado em produção** — o usuário vai rodar e reportar o que acontecer.

- **Se falhar** (ffmpeg não conseguir abrir o pipe, travar, produzir vídeo corrompido, ou reproduzir a mesma incompatibilidade de lib da 1ª tentativa): **reverter é a opção segura** — desfazer as 5 mudanças acima volta exatamente ao comportamento anterior (mp4v sempre, reencode completo sempre em `__convert_video`), que é conhecido e estável. Não tentar "consertar" o pipe às pressas sem entender a causa raiz do erro — trazer o erro completo (stderr do ffmpeg, que agora fica anexado à `RuntimeError`) para análise antes de decidir entre corrigir o pipe ou reverter.
- **Se funcionar**: valida a hipótese de que a incompatibilidade da 1ª tentativa era específica do `cv2.VideoWriter` (build do OpenCV sem suporte a H.264), não do ffmpeg em si — e representa o maior ganho de performance ainda disponível no pipeline (elimina um encode CPU completo por vídeo).
- **Alternativas a considerar caso esta 2ª tentativa também falhe**: (a) manter o mp4v mas trocar a compressão/qualidade dele para algo mais rápido de decodificar depois; (b) investigar se dá para usar `ffmpeg-python` (já importado no projeto como `ffmpeg`) para abrir o pipe em vez de `subprocess` direto, o que pode simplificar o tratamento de erro; (c) aceitar o encode duplo como custo fixo e focar esforço de performance nos achados #9/#10 (I/O de cache) em vez de tentar uma 3ª via para o encode.

### 9. `reface_with_light_cache` recalcula hash e reabre o vídeo mesmo com frame-cache disponível; contagem de hit rate é enganosa

- **Arquivo:linha**: `refacer.py:429` (`_compute_video_hash`) e `436-440`; lookup duplicado em `refacer.py:489-495`
- **Status**: ainda presente, não corrigido.
- **Problema**: mesmo quando os frames já estão decodificados em RAM (`cached_frames`), o código recomputa o hash (leitura de 2MB do disco) e abre um `cv2.VideoCapture` só para ler `fps/width/height` (linhas 436-440, só evitado quando `cached_frames is not None`, mas o hash em si — linha 429 — sempre é recomputado a cada chamada). Além disso, há um lookup duplicado de cache por frame só para contar hit/miss (linhas 492-495) que sempre resulta em "hit" porque acabou de inserir na linha anterior — a métrica de hit rate impressa (linha 526) fica sempre próxima de 100%, mesmo em cache miss real.
- **Impacto**: no fluxo multi-rosto (um job por rosto, mesmo vídeo), hash é refeito a cada job — I/O e CPU desperdiçados sob GIL. A métrica de hit rate incorreta também mascara se o light cache está de fato funcionando.
- **Sugestão**: cachear `fps/width/height` junto com os frames decodificados; computar o hash uma vez fora da função e reaproveitar entre jobs; remover o lookup redundante de cache (basta capturar se `__get_faces_with_light_cache` fez hit/miss internamente, sem lookup extra). Nota: o fluxo mais novo (`analyze_video_in_memory` + `reface_with_precomputed`, acionado em `app.py` quando há múltiplos jobs) já evita esses problemas — é o caminho a favorecer; vale confirmar se `reface_with_light_cache` ainda é exercitado na prática (só quando `use_cache=True` e há exatamente 1 job, ou seja, 1 único rosto-alvo configurado).

### 10. `reface_with_cache` grava/lê 1 arquivo `.npz` por frame — I/O pulverizado

- **Arquivo:linha**: escrita em `refacer.py:897-912` (`analyze_target_video`), leitura em `refacer.py:1054-1061` (`reface_with_cache`)
- **Status**: ainda presente, não corrigido.
- **Problema**: o loop acumula 100 frames (`batch_size = 100`, linha 863) e salva cada um em seu próprio `.npz` (`np.savez_compressed`) — mesmo número de arquivos que sem batching, com compressão zlib (CPU-bound) por arquivo.
- **Impacto**: usado só para vídeos > 30s com `use_cache` ligado (ramo `duration_seconds < 60` ou maior em `refacer.py:1301-1322`). Em CPU limitada, comprimir e abrir/fechar milhares de arquivos pequenos é caro, e filesystems de Colab/Lightning costumam ser lentos.
- **Sugestão**: consolidar em poucos arquivos (um `.npz` por bloco de 100 frames, ou um único arquivo com arrays empilhados + índice) e trocar `savez_compressed` por `savez` — bboxes/kpss são minúsculos, a compressão zlib não compensa. Vale confirmar antes se esse caminho (vídeos > 30-60s com cache ligado) ainda é usado na prática, já que jobs múltiplos sobre o mesmo vídeo agora preferem `analyze_video_in_memory`.

---

## Parte 3 — Qualidade (custo de performance baixo/desprezível, do review anterior)

### 11. CodeFormer aplicado só em imagem/TIFF, nunca em vídeo — decisão correta, mantê-la

- **Arquivo:linha**: `refacer.py:1471` e `1449` (imagem/TIFF chamam `enhance_image`/`enhance_image_memory`); nenhum caminho de vídeo chama enhancement.
- **Observação**: rodar CodeFormer por frame de vídeo seria caro demais (modelo de restauração + detecção própria por frame) — não fazer isso em vídeo é a decisão certa dado o gargalo de CPU. Confirmação de boa prática, não um problema.
- **Ajuste barato disponível**: o parâmetro `w=0.5` (fidelidade vs. qualidade) em `codeformer_wrapper.py:59` é fixo; para imagens, um `w` mais alto (~0.7) preserva mais identidade/fidelidade. Custo de mudança: zero, mas afeta apenas o modo imagem, não vídeo.
- **Sugestão**: manter vídeo sem enhancement. Opcionalmente expor `w` como slider no modo imagem.

### 12. ~~Limitação conhecida do "Reface Ratio" com rostos ocluídos~~ — **implementado**: máscara oval opt-in, boca→queixo

- **Arquivo:linha**: `refacer.py` — `_partial_face_blend`, `_rect_cutoff_mask`, `_mouth_chin_oval_mask` (flag `partial_blend_shape`); checkbox "Oval Mask" em `app.py` (aba Video Mode, ao lado do slider "Reface Ratio").
- **Discutido e descartado no review anterior**: cogitou-se trocar o corte reto por máscara oval/elíptica acompanhando **todo o contorno do rosto** (via keypoints já detectados), para bochecha retraída onde a linha reta corta visivelmente o contorno. Descartado como comportamento **padrão** porque uma elipse larga não sabe distinguir pele de oclusão real no queixo/bochecha (mão, microfone, sombra dura) — nesses casos borraria/misturaria o objeto com o frame anterior em vez de preservá-lo, pior que o corte reto (previsível: tudo abaixo da linha fica 100% preservado).
- **Ideia revisada que foi implementada**: em vez de uma elipse larga seguindo todo o contorno, uma elipse **estreita e vertical**, indo do lábio superior até o queixo, sem se aproximar das laterais — bochechas ficam fora da elipse por definição geométrica, não por heurística de oclusão. Isso reduz bastante a área de risco (só sobra oclusão que caia dentro da própria faixa boca-queixo, ex. mão tampando a boca), mantendo o ganho de contorno mais natural no centro do rosto.
- **Como foi calculada**: sem landmark de queixo disponível (o detector só dá 5 keypoints: 2 olhos, nariz, 2 cantos de boca), o topo da elipse usa o ponto médio dos 2 kps de canto de boca, e a base usa `y2` do bbox como aproximação do queixo; a largura vem da distância entre os cantos de boca. Recalculado por frame (acompanha movimento de cabeça); cai automaticamente no corte reto (`_rect_cutoff_mask`) se `face.kps` vier `None`.
- **Relacionado ao achado #5 desta revisão** (pose de cabeça): a elipse não resolve o problema de #5 (pose extrema ainda distorce a geometria de origem dos keypoints), mas reduz a dependência da orientação do bbox por não usar mais uma linha reta fixa.
- **Pendente**: validar visualmente em vídeo real com oclusão na região boca/queixo (mão, microfone) para confirmar que o comportamento nesse caso residual é aceitável antes de considerar promover a opção a padrão.

---

## Parte 4 — Achados novos do segundo passe (não cobertos pelos reviews anteriores)

Verificados linha a linha contra o código atual em 2026-07-16. Os achados #13 e #14 afetam diretamente a 2ª tentativa de encode via pipe (achado #8) e devem ser considerados **antes** do teste em produção.

### 13. O fallback do `_open_video_writer` quase nunca dispara — falhas reais do ffmpeg estouram no meio do processamento, sem fallback

- **Arquivo:linha**: `refacer.py:690-700` (`_open_video_writer`), `refacer.py:91-108` (`write`/`release` de `_FfmpegVideoWriter`)
- **Problema**: o fallback para `cv2.VideoWriter`+mp4v só é acionado se a **construção** de `_FfmpegVideoWriter` levantar exceção — e a construção só falha se o binário `ffmpeg` não existir (o `subprocess.Popen` abre com sucesso mesmo com encoder inválido, resolução incompatível ou GPU indisponível; o erro do ffmpeg só aparece depois, ao processar o primeiro frame). Nesses casos reais de falha, o erro surge como `RuntimeError` dentro de `write()` (BrokenPipeError quando o pipe enche, frame Full HD tem ~6MB) ou em `release()` — **depois** que o writer já foi retornado como "bom".
- **Impacto**: (a) o fallback desenhado nunca protege contra os modos de falha plausíveis; (b) como os caminhos de processamento acumulam até `batch_size` frames (até 1000) antes do primeiro `output.write`, uma falha do encoder é detectada só depois de descartar até 1000 frames de trabalho de GPU; (c) a `RuntimeError` propaga até o loop de jobs em `app.py:316` e aborta os jobs restantes (agrava o achado #6). O único cenário em que o fallback funciona é `ffmpeg` ausente do PATH — que nunca acontece, pois `__check_encoders` já rodou no startup.
- **Sugestão**: validar o pipe de verdade antes de devolver o writer — ex.: escrever 1 frame preto de teste num `_FfmpegVideoWriter` descartável com os mesmos parâmetros (ou checar `process.poll()` após um pequeno write inicial) dentro do `try` de `_open_video_writer`; se falhar, cair para mp4v ali mesmo. Alternativa mais simples: capturar a `RuntimeError` de `write()`/`release()` no chamador e reprocessar o vídeo com o writer mp4v (custo: reprocessa 1 vídeo, mas não aborta o lote).

### 14. Pipe ffmpeg com `-pix_fmt yuv420p` falha em vídeos com dimensão ímpar — o mp4v antigo aceitava

- **Arquivo:linha**: `refacer.py:75` (`command += ['-pix_fmt', 'yuv420p', output_path]`)
- **Problema**: H.264 com `yuv420p` exige largura e altura pares (subsampling 4:2:0). O `cv2.VideoWriter` mp4v aceitava qualquer dimensão; o pipe novo faz o ffmpeg abortar com "width/height not divisible by 2" em vídeos com dimensão ímpar (crops manuais, vídeos de celular processados por outras ferramentas). Combinado com o achado #13, isso não cai no fallback — vira `RuntimeError` no meio do job.
- **Impacto**: regressão silenciosa de compatibilidade da 2ª tentativa de encode; é exatamente o tipo de "incompatibilidade" que pode ser confundida com a da 1ª tentativa ao testar.
- **Sugestão**: no `_open_video_writer`, se `frame_width % 2 != 0 or frame_height % 2 != 0`, ou usar o fallback mp4v direto, ou adicionar `-vf pad=ceil(iw/2)*2:ceil(ih/2)*2` ao comando do pipe (custo zero para vídeos já pares).

### 15. Preview + `precomputed`: vídeo sai em duração cheia com só 1 a cada 10 frames trocados, e a estimativa de RAM fica 10× menor que o real

- **Arquivo:linha**: `refacer.py:383-400` (`analyze_video_in_memory` — frames pulados são anexados como `(frame, None, None, None)`), `refacer.py:436-449` (`reface_with_precomputed` escreve o frame original quando `bboxes is None`), `refacer.py:356-359` (estimativa usa `range(0, total, skip_rate)`)
- **Problema**: nos outros caminhos, preview **descarta** os frames pulados (vídeo final curto, todos os frames trocados). No caminho `precomputed`, os frames pulados são mantidos e escritos **sem swap** — o preview sai com a duração original, com 9 de cada 10 frames mostrando o rosto original (flicker constante), e o encode roda sobre o vídeo inteiro (nenhum ganho de tempo no encode). Além disso, a estimativa inicial de RAM (`decoded_frame_count`) divide por `skip_rate`, mas `entries` acumula todos os frames — a checagem prévia subestima o consumo em 10× no preview (o guard de `running_bytes` nas linhas 376-381 ainda protege, mas só depois de gastar o decode).
- **Impacto**: qualquer uso de "Preview Generation" + "Enable Cache" + 2+ jobs produz um preview visivelmente quebrado e mais lento do que o preview dos outros caminhos.
- **Sugestão**: em `analyze_video_in_memory`, quando `preview=True`, simplesmente **não** anexar os frames pulados a `entries` (igual aos outros caminhos) — corrige o flicker, a duração, o tempo de encode e a estimativa de RAM de uma vez.

### 16. Em "Faces By Match", mesmo com um único rosto-alvo, o escolhido é o rosto mais à direita acima do threshold — não o mais similar

- **Arquivo:linha**: `refacer.py:1048-1059` (`_apply_swaps`, ramo `else`)
- **Problema**: complementa o achado #1, que só tratou do empate entre múltiplos `rep_face`. O loop interno percorre os rostos detectados da direita para a esquerda (`range(len(faces)-1, -1, -1)`, lista ordenada por `bbox[0]`) e aceita o **primeiro** com `sim >= threshold` — não o de maior similaridade. Com o threshold padrão de 0.2 (baixo), num frame com 2+ pessoas é comum mais de um rosto passar do threshold; o escolhido é sempre o mais à direita, mesmo que outro rosto tenha similaridade muito maior.
- **Impacto**: swap na pessoa errada em cenas com múltiplas pessoas, mesmo no caso simples de 1 rosto-alvo — e alternância entre frames conforme os rostos se movem horizontalmente (outra fonte do "flicker" além dos achados #1/#2).
- **Sugestão**: dentro do loop de `rep_face`, calcular `sim` para **todos** os rostos restantes e trocar o `argmax` (se `>= threshold`), em vez de aceitar o primeiro. É a versão mínima da correção do achado #1 e pode ser feita junto (a matriz global de similaridade cobre os dois).

### 17. O caminho `precomputed` está preso ao checkbox "Enable Cache" — multi-job sem o checkbox refaz decode+detecção+embedding N vezes

- **Arquivo:linha**: `app.py:313` (`if use_cache and len(jobs) > 1:`)
- **Problema**: `analyze_video_in_memory` não é um cache persistente — é uma variável local que vive só durante o `run()`, sem hash, sem estado entre execuções. Condicioná-lo ao checkbox "Enable Cache (Faster subsequent runs)" (que descreve outra coisa: cache entre execuções) faz com que o cenário multi-rosto com o checkbox desligado — o default da UI — pague N passadas completas de decode + detecção + embedding sobre o mesmo vídeo.
- **Impacto**: com 3 rostos-alvo e checkbox desligado (padrão), o custo de análise triplica sem motivo; é exatamente o desperdício que o `precomputed` foi criado para eliminar.
- **Sugestão**: trocar a condição para `len(jobs) > 1` apenas (o teto de RAM `in_memory_analysis_max_mb` já protege contra vídeos grandes, com fallback automático para o caminho por-job). O checkbox continua controlando os caches persistentes (`use_cache` no `reface()`).

### 18. `first_face` / `process_first_face` são vestigiais — flag calculada em 4 lugares para escolher entre duas funções idênticas

- **Arquivo:linha**: `refacer.py:1239-1240` (`process_first_face` só chama `process_faces`), flag atribuída em `refacer.py:422, 482, 1078, 1395` e consumida apenas em `refacer.py:1133` e `1253` — ambos escolhendo entre as duas funções idênticas
- **Problema**: `self.first_face` não altera nenhum comportamento hoje (`_apply_swaps` nunca a consulta). No modo "Single Face", o ramo `disable_similarity` com 1 destino troca **todos** os rostos detectados do frame pelo mesmo destino — se a intenção histórica de "first face" era trocar só um rosto, isso se perdeu; se o comportamento atual é o desejado, a flag e o método são código morto que sugere uma distinção inexistente.
- **Impacto**: sem efeito em runtime; custo é de manutenção/leitura (qualquer correção futura de matching precisa primeiro descobrir que a flag não faz nada). Confirmar com o usuário qual comportamento "Single Face" deve ter num frame com várias pessoas.
- **Sugestão**: remover `process_first_face` e todas as atribuições de `self.first_face`, ou reimplementar a semântica de "só o rosto principal" (ex.: maior bbox) se ela for desejada.

### 19. Rosto inclinado/de perfil ou deitado perde integridade no swap — limitação estrutural do alinhamento, não do `refacer.py`

- **Arquivo:linha**: `recognition/face_align.py:42` (`estimate_norm`) e `:71` (`norm_crop`), `recognition/arcface_onnx.py:65`, `refacer.py:1018` (`face_align.estimate_norm(kps, image_size=112)[0]`) — todas as chamadas usam `mode='arcface'` (implícito ou explícito), nunca o modo alternativo.
- **Causa raiz**: o alinhamento facial usado antes do swap é uma `SimilarityTransform` (só rotação + escala uniforme + translação — 4 graus de liberdade, sem shear/perspectiva) contra um único template frontal fixo (`arcface_src`, `face_align.py:31-36`). Existem no mesmo arquivo 5 templates de pose (`src1..src5`, `face_align.py:5-28` — perfil esquerdo, meio-perfil esquerdo, frontal, meio-perfil direito, perfil direito) que o modo `'non-arcface'` escolheria automaticamente pelo menor erro residual, mas esse modo está **morto**: nenhuma chamada no projeto (nem na lib `insightface` 0.7.3 da qual o arquivo foi herdado) usa `mode` diferente de `'arcface'`.
- **Por que "perde integridade" em pose lateral/deitada**: nenhuma transformação 2D (mesmo escolhendo o melhor entre os 5 templates) reconstrói a perspectiva/profundidade 3D perdida quando a cabeça gira fora do plano da imagem (yaw/pitch). Roll (cabeça "deitada", rotação *no plano* da imagem) já é bem tratado hoje pela `SimilarityTransform` — o problema real é yaw/pitch acentuado (virar de lado, inclinar para frente/trás), onde o crop alinhado fica geometricamente inconsistente e o swap final sai distorcido/borrado.
- **Agravante**: o `INSwapper.get()` que faz o swap propriamente dito (código de terceiros da lib `insightface`, não deste repositório) faz seu **próprio** realinhamento interno, sempre no template frontal fixo — então mesmo ativando o modo multi-template em `face_align.py`/`arcface_onnx.py` (usado hoje só para o embedding de reconhecimento), o crop 128×128 usado dentro do swap em si continuaria frontal. Mudar isso de verdade exigiria monkeypatch ou fork do `INSwapper`, não é um parâmetro exposto.
- **Custo de ativar o modo multi-template**: praticamente nulo — escolher o menor erro entre 5 templates é resolver uma `SimilarityTransform` sobre 5 pontos 5 vezes em vez de 1 (SVD pequena, O(1) na prática), irrelevante perto do custo de qualquer inferência de rede (detecção, embedding, swap). O `warpAffine` final continua rodando uma única vez, só a escolha do melhor `M` seria repetida.
- **Resolveria o problema todo?** Não — só mitigaria pose *moderada* (rosto de 3/4), melhorando o ajuste de escala/proporção do crop. Pose realmente extrema (perfil quase total, cabeça bem inclinada) continuaria distorcida, por limite físico de projeção 3D→2D, e mesmo essa mitigação parcial não teria efeito no crop interno do `INSwapper` sem fork/monkeypatch.
- **Outras limitações relacionadas, sem solução barata**: resolução fixa 128×128 do modelo `inswapper_128.onnx` (menos "área útil" de rosto cabe no quadrado quando há escorço de pose); ausência de correção de iluminação/bordas pretas no crop alinhado (`cv2.warpAffine` com `borderValue=0.0`, comum quando parte da cabeça sai do frame em pose extrema); nenhuma estimativa explícita de pose (yaw/pitch/roll) em nenhum ponto do pipeline hoje.
- **Sugestão**: não implementar agora — o ganho é parcial (só pose moderada) e o esforço de fazer diferença real (fork/monkeypatch do `INSwapper` de terceiros) é desproporcional ao benefício. Registrar como limitação conhecida. Se o usuário quiser revisitar, o menor primeiro passo de custo quase zero seria ativar `mode='non-arcface'` no crop de reconhecimento (`arcface_onnx.py:65`) só para validar se a mitigação parcial (pose 3/4) é perceptível na prática antes de considerar o fork do swap em si.

### 20. Menores (registrar, baixa prioridade)

- **`__try_ffmpeg_encoder` deixa `testsrc.mp4` no diretório de trabalho** (`refacer.py:1581-1587`): o teste de encoder do startup grava um mp4 de 1s no CWD e nunca remove. Lixo pequeno, mas em Colab o arquivo aparece na raiz do notebook a cada start. Usar `tempfile` ou `-f null -`.
- **Paridade de qualidade do pipe não conferida**: o pipe novo omite `-b:v` quando o bitrate é `'0'` (`refacer.py:73-74`), enquanto o `__convert_video` antigo passava `-b:v 0` explicitamente. Para `h264_nvenc`, sem nenhum controle de rate o default do ffmpeg é um bitrate fixo baixo (~2Mbps), que em 1080p é visivelmente pior que o resultado do caminho antigo. Ao validar a 2ª tentativa, comparar qualidade além de "funcionou"; se necessário, adicionar `-cq`/`-preset` ao comando do pipe.
- **Intermediário nunca é apagado quando há mux/reencode** (`refacer.py:1478-1494`): quando `__convert_video` gera `*_c.mp4`, o arquivo intermediário permanece em `output/` — 2× o disco por vídeo processado. Pré-existente (não é regressão do pipe), mas em disco pequeno de Colab acumula rápido.

---

## Boas práticas já presentes (manter)

- Priorização de `h264_nvenc` em `__check_encoders` (usa a GPU quando disponível).
- `use_num_cpus=2` e `intra_op_num_threads=1` no modo CUDA — evita oversubscription de threads no ONNX Runtime.
- Caminho `analyze_video_in_memory` + `reface_with_precomputed`: decodifica/detecta/computa embeddings uma única vez e passa os dados por variável local, evitando cache com hash e contabilidade de memória por frame — é o caminho preferido quando há múltiplos jobs sobre o mesmo vídeo.
- Extração de `_apply_swaps`/`_process_faces_with_cached_data`: centralizar matching+swap+blend num único método compartilhado por todos os caminhos de cache foi a decisão certa — os achados #1, #2, #3 e #5 deste review agora só precisam de correção em um lugar, não em quatro.
- Separação clara `disable_similarity` vs `multiple_faces_mode` vs "Faces By Match" (`refacer.py:944-986`): os três caminhos de matching não se sobrepõem incorretamente — cada modo da UI mapeia para exatamente um ramo de `_apply_swaps`.
- `compute_sim` via cosseno padrão (`recognition/arcface_onnx.py:69-74`): implementação padrão e correta para embeddings ArcFace.
- Threshold configurável por rosto-alvo (`thresholds_video` em `app.py`, um slider por Face #k): dá controle fino por rosto — é a mitigação manual disponível hoje para o achado #1 (subir o threshold do rosto mais "confundível").
- Decisão de não rodar CodeFormer no pipeline de vídeo (custo alto demais para o ganho, dado o hardware).
