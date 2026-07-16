# Review — Performance e Qualidade do Processamento de Vídeo

Escopo: `refacer.py` (pipeline completo de vídeo) e `app.py` (aba de vídeo/Gradio).

Contexto de hardware assumido: GPU Tesla T4 (16GB VRAM), 15GB de RAM, **CPU muito limitada** (Google Colab free tier / Lightning AI free tier — poucos vCPUs). GPU e RAM são recursos relativamente folgados; CPU é o gargalo real.

Critério aplicado: achados de qualidade só entram se o ganho compensar um custo de performance baixo ou desprezível nesse hardware. Sugestões de qualidade que exigiriam bem mais CPU/GPU/tempo (upscaling pesado, enhancement por frame, multi-escala em todo frame) foram descartadas propositalmente.

## Prioridade sugerida

1. **#8 / #12** — Encode único na GPU (ganho de performance *e* qualidade juntos)
2. **#7** — Cachear embeddings no fluxo multi-job (elimina inferência redundante na T4)
3. **#1 / #3** — Eliminar contenção de threads e `gc.collect()` supérfluo
4. **#9** — Teto de RAM em `analyze_video_in_memory` para evitar OOM no Colab

---

## Performance

### 1. `ThreadPoolExecutor` no caminho não-cacheado é inútil (ou prejudicial) em CPU limitada + CUDA

- **Arquivo:linha**: `refacer.py:1145-1156` (`reface_group`), chamado em `refacer.py:1306` e `1314`
- **Problema**: `reface_group` distribui `process_faces`/`process_first_face` num `ThreadPoolExecutor(max_workers=self.use_num_cpus)`. Cada tarefa faz detecção, embedding e swap via ONNX Runtime configurado com `intra_op_num_threads=1` no modo CUDA (`refacer.py:649-651`) — os três modelos compartilham a mesma GPU/sessão. A orquestração Python (montagem de `Face`, `sorted`, `compute_sim`, blending numpy) roda sob GIL.
- **Impacto**: com poucos vCPUs, jogar threads contra uma única sessão ONNX serializada não paraleliza inferência — só adiciona contenção de GIL e overhead de agendamento. O trabalho Python puro não escala com threads. Os caminhos cacheados (`reface_with_light_cache`, `reface_with_precomputed`) já processam serialmente e são mais coerentes com esse hardware.
- **Sugestão**: no modo CUDA/T4, processar frames serialmente em vez de usar o pool. Se quiser manter paralelismo, ele só compensa para sobrepor *decode* (I/O, libera GIL) com inferência GPU — nesse caso, um pool dedicado só à leitura de frames, não um pool que replica todo o pipeline. O ganho mais seguro e barato é simplesmente serializar.

### 2. `_calculate_optimal_batch_size` não controla batch de inferência — só o tamanho do flush do writer

- **Arquivo:linha**: `refacer.py:508-546`, usada em `refacer.py:335, 422, 999, 1292`
- **Problema**: a função calcula `batch_size` a partir de VRAM/FPS, mas a inferência é frame a frame — `batch_size` só controla de quantos em quantos frames o código acumula em RAM antes de um `output.write` em lote. Não há batch de inferência real, e o `VideoWriter` já bufferiza sozinho.
- **Impacto**: os prints (`[BATCH] Dynamic batch size...VRAM`) sugerem otimização de GPU que não existe. Em vídeos maiores, segurar até 1000 frames Full HD em RAM (~6GB) antes de escrever pressiona os 15GB sem ganho de velocidade. O efeito colateral real é a frequência de `gc.collect()` (achado #3).
- **Sugestão**: para ganho real de GPU seria preciso batching de inferência de verdade (empilhar N crops num único tensor para `inswapper`/`arcface`). Caso não valha o esforço agora, simplificar: escrever cada frame direto no writer e remover a lista intermediária, eliminando custo de RAM e a falsa sensação de otimização.

### 3. `gc.collect()` explícito em loop quente

- **Arquivo:linha**: `refacer.py:364, 474, 1030, 1062, 1308`
- **Problema**: a cada flush de batch é chamado `gc.collect()` manualmente — uma varredura completa e síncrona de todos os objetos rastreados.
- **Impacto**: dispara varreduras completas de GC repetidamente durante o processamento. Frames grandes (numpy arrays) já são liberados por contagem de referência ao esvaziar a lista — não por ciclo de GC. Em CPU muito limitada, cada `gc.collect()` compete por tempo de CPU com o que realmente importa.
- **Sugestão**: remover as chamadas. Se houver receio de fragmentação de VRAM, a ferramenta certa seria `torch.cuda.empty_cache()` (não `gc.collect()`), e mesmo assim não por-batch.

### 4. Profiling sempre ligado, custando overhead por-rosto/por-frame

- **Arquivo:linha**: `refacer.py:75` (`profiling_enabled = True`), `_profile_start`/`_profile_end` em `93-104`, invocados em `233-235, 242-244, 259-261, 297-299, 350-352, 1111-1113, 1124-1126, 1140-1142`
- **Problema**: `profiling_enabled` está hardcoded como `True`. Para cada rosto de cada frame há 2-3 pares `time.time()` + atualização de `defaultdict`.
- **Impacto**: em vídeos com milhares de frames e múltiplos rostos isso vira dezenas de milhares de chamadas + acessos a dict sob GIL — 100% overhead de instrumentação, sem benefício de resultado.
- **Sugestão**: deixar `profiling_enabled = False` por padrão, expondo como flag/env var para debug. Como as funções já fazem early-return quando desligado, o custo cai a zero sem remover código.

### 5. `reface_with_cache` grava/lê 1 arquivo `.npz` por frame — I/O pulverizado

- **Arquivo:linha**: escrita em `refacer.py:855-872` (`analyze_target_video`), leitura em `refacer.py:1014-1021` (`reface_with_cache`)
- **Problema**: o comentário promete "batch de I/O eficiente", mas o loop acumula 100 frames e salva cada um em seu próprio `.npz` (`np.savez_compressed`) — mesmo número de arquivos, com compressão zlib (CPU-bound) por arquivo.
- **Impacto**: usado só para vídeos > 30s com `use_cache` ligado. Em CPU limitada, comprimir e abrir/fechar milhares de arquivos pequenos é caro, e filesystems de Colab/Lightning costumam ser lentos.
- **Sugestão**: consolidar em poucos arquivos (um `.npz` por bloco de 100 frames, ou um único arquivo com arrays empilhados + índice) e trocar `savez_compressed` por `savez` — bboxes/kpss são minúsculos, a compressão zlib não compensa. Vale confirmar antes se esse caminho (vídeos longos) ainda é usado na prática.

### 6. `reface_with_light_cache` recalcula hash e reabre o vídeo mesmo com frame-cache disponível

- **Arquivo:linha**: `refacer.py:394` (`_compute_video_hash`) e `401-405`
- **Problema**: mesmo quando os frames já estão decodificados em RAM, o código recomputa o hash (leitura de 2MB do disco) e abre um `cv2.VideoCapture` só para ler `fps/width/height`. Além disso, há um lookup duplicado de cache por frame só para contar hit/miss — que sempre resulta em "hit" porque acabou de inserir, tornando a métrica de hit rate enganosa.
- **Impacto**: no fluxo multi-rosto (um job por rosto, mesmo vídeo), hash e capture são refeitos a cada job — I/O e CPU desperdiçados sob GIL.
- **Sugestão**: cachear `fps/width/height` junto com os frames decodificados; computar o hash uma vez e reaproveitar; remover o lookup redundante de cache. Nota: o fluxo mais novo (`analyze_video_in_memory` + `reface_with_precomputed`, acionado em `app.py`) já evita esses problemas — é o caminho a favorecer.

### 7. Embedding recomputado a cada job no fluxo multi-rosto

- **Arquivo:linha**: `refacer.py:296-302` (analyze) vs `refacer.py:350-352` (reface_with_precomputed)
- **Problema**: `analyze_video_in_memory` já detecta rostos (bboxes/kpss) uma única vez por frame. Mas o **embedding** (`rec_app.get`) depende só do frame, não da face de origem — e ainda assim é recomputado a cada job/rosto-alvo que reutiliza o mesmo vídeo.
- **Impacto**: com N rostos-alvo configurados sobre o mesmo vídeo (exatamente o cenário que motivou o `precomputed`), o embedding — uma inferência GPU por rosto por frame — é recalculado N vezes.
- **Sugestão**: computar e armazenar os embeddings junto com bboxes/kpss dentro de `analyze_video_in_memory` (custa 1x, na única passada de detecção); no replay, reconstruir `Face` já com `embedding` preenchido em vez de chamar `rec_app.get` de novo. Custo extra de RAM é trivial (512 floats por rosto por frame); o ganho é eliminar N-1 passadas de inferência de embedding — ganho direto de GPU/CPU exatamente no caso que o `precomputed` foi criado para otimizar.

### 8. Encode intermediário em `mp4v` força recodificação completa em `__convert_video`

- **Arquivo:linha**: escrita mp4v em `refacer.py:332-333, 418-419, 989-994, 1288-1289`; `__convert_video` em `refacer.py:1352-1362`
- **Problema**: todos os caminhos escrevem o resultado com `cv2.VideoWriter` em `mp4v` (MPEG-4 Part 2). Depois, `__convert_video` só recodifica para H.264/nvenc **quando há áudio**. Quando há áudio, isso significa decodificar o mp4v inteiro e recomprimir em H.264 — dois encodes de vídeo inteiro onde bastaria um.
- **Impacto**: a escrita inicial em mp4v via OpenCV já é um encode CPU-bound do vídeo inteiro; o reencode subsequente (quando há áudio) repete o trabalho. Em CPU limitada isso é um custo significativo e evitável.
- **Sugestão**: evitar o writer mp4v do OpenCV — escrever frames crus (rawvideo) num pipe para o ffmpeg com `h264_nvenc` fazendo o único encode na GPU, já muxando o áudio original (`-c:a copy`) no mesmo comando. Isso elimina o encode CPU do mp4v e a recodificação dupla — provavelmente o maior ganho de performance disponível para esse hardware, pois move todo o encode para a GPU numa única passada.

### 9. `analyze_video_in_memory` mantém todos os frames decodificados em RAM sem teto

- **Arquivo:linha**: `refacer.py:288-314` (lista `entries` com `(frame, bboxes, kpss)`)
- **Problema**: diferente de `_store_cached_frames` (que já tem limite de 4GB), `entries` acumula o frame decodificado inteiro de cada posição do vídeo sem limite algum. Um vídeo de 60s a 30fps Full HD já chega a ~11GB.
- **Impacto**: com 15GB de RAM, um vídeo Full HD de ~1 min já se aproxima do limite; vídeos maiores estouram a RAM e travam — cenário comum em Colab free. O gatilho (`use_cache` + múltiplos jobs) é acionado justamente quando o usuário troca vários rostos, caso em que o vídeo pode ser mais longo.
- **Sugestão**: aplicar o mesmo teto de RAM que `_store_cached_frames` já usa (estimar tamanho; se exceder, cair para o caminho por-job que redecodifica) ou limitar por número de frames/duração. Em vídeos grandes perde-se a otimização de decode único, mas evita OOM — que é bem pior. Baixo custo de implementação, alto valor de robustez.

---

## Qualidade (custo de performance baixo/desprezível)

### 10. CodeFormer aplicado só em imagem/TIFF, nunca em vídeo — decisão correta, mantê-la

- **Arquivo:linha**: `refacer.py:1393` e `1415` (imagem/TIFF chamam `enhance_image`/`enhance_image_memory`); nenhum caminho de vídeo chama enhancement
- **Observação**: rodar CodeFormer por frame de vídeo seria caro demais (modelo de restauração + detecção própria por frame) — não fazer isso em vídeo é a decisão certa dado o gargalo de CPU. É uma confirmação de boa prática, não um problema.
- **Ajuste barato disponível**: o parâmetro `w=0.5` (fidelidade vs. qualidade) em `codeformer_wrapper.py:59` é fixo; para imagens, um `w` mais alto (~0.7) preserva mais identidade/fidelidade. Custo de mudança: zero (é só um número), mas afeta apenas o modo imagem, não vídeo.
- **Sugestão**: manter vídeo sem enhancement. Opcionalmente expor `w` como slider no modo imagem.

### 11. Transição de blending parcial pode ser suavizada sem custo adicional — **implementado**

- **Arquivo:linha**: `refacer.py:582-619` (`_partial_face_blend`), `transition = 40`
- **Problema**: a transição usava `np.linspace` linear numa faixa fixa de 40px. Uma curva suave (smoothstep) na mesma janela custa exatamente o mesmo, pois já é uma operação vetorizada numpy sobre uma região pequena (só a área do rosto, só quando `partial_reface_ratio > 0`).
- **Impacto**: transição linear podia deixar uma "borda" visível na altura de corte; smoothstep reduz esse artefato com custo de CPU desprezível.
- **Status**: aplicado — `alpha` agora usa smoothstep (`3t² − 2t³`) na mesma janela `transition`, sem mudar a geometria do corte (ainda é uma linha reta horizontal, só a curva de mistura ficou mais suave nas pontas).
- **Discutido e descartado**: cogitou-se trocar o corte reto por uma máscara oval/elíptica acompanhando o formato do rosto (via os keypoints já detectados), para o caso de rostos com bochecha retraída onde a linha reta corta visivelmente o contorno. Descartado como comportamento padrão porque uma elipse fixa não sabe distinguir pele de uma oclusão real no queixo (mão, microfone, sombra dura) — nesses casos ela borraria/misturaria o objeto com o frame anterior em vez de preservá-lo, o que é pior que o corte reto atual (previsível: tudo abaixo da linha fica 100% preservado). Resolver isso direito exigiria segmentação de oclusão por frame, que é cara demais para o hardware-alvo.
- **Sugestão futura (não implementada)**: expor a máscara oval como opção **opt-in via flag** (ex: `partial_blend_shape="oval"` vs. o padrão atual `"rect"`), sem tocar no comportamento padrão, feita com base nos keypoints já calculados (sem inferência extra) e sem chegar até o queixo (encerrar a elipse antes da região mais provável de oclusão) — para permitir testes controlados antes de considerar virar padrão.

### 12. Vídeos sem áudio saem no encode mp4v (qualidade inferior), enquanto vídeos com áudio saem em H.264

- **Arquivo:linha**: `refacer.py:1359-1360` (ramo `else`: mantém o `output_video_path` original em mp4v)
- **Problema**: relacionado ao achado #8 — quando o vídeo não tem áudio, o resultado final permanece no mp4v do OpenCV (MPEG-4 Part 2, bitrate padrão), com compressão/qualidade inferior a H.264. Só vídeos com áudio passam pelo reencode H.264/nvenc melhor.
- **Impacto**: vídeos sem áudio saem com qualidade visivelmente pior, sem indicação clara do porquê. Resolver o achado #8 (pipe direto para nvenc) resolve isto ao mesmo tempo: um único encode, na GPU, para todos os vídeos — sem custo extra de CPU.
- **Sugestão**: sempre passar o resultado pelo encoder H.264/nvenc, com ou sem áudio, idealmente eliminando o mp4v intermediário conforme #8. Como o encoder já roda na T4, o custo de CPU adicional é ~nulo e a qualidade sobe de forma consistente.

---

## Boas práticas já presentes (manter)

- Priorização de `h264_nvenc` em `__check_encoders` (usa a GPU quando disponível).
- `use_num_cpus=2` e `intra_op_num_threads=1` no modo CUDA — evita oversubscription de threads no ONNX Runtime.
- Caminho `analyze_video_in_memory` + `reface_with_precomputed`: decodifica/detecta uma única vez e passa os dados por variável local, evitando cache com hash e contabilidade de memória por frame.
- Decisão de não rodar CodeFormer no pipeline de vídeo (custo alto demais para o ganho, dado o hardware).
