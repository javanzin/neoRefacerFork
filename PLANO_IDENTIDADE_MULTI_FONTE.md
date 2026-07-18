# Plano Técnico — Aba "Criar Identidade" (multi-imagem / multi-vídeo → perfil facial reutilizável)

Status: **planejamento apenas — nenhum código foi alterado**.
Escopo confirmado com o usuário: a identidade agregada será consumida **somente no modo Vídeo**. O modo Imagem/GIF/TIFF com foto única de destino **não muda**.
Projeto: NeoRefacer (app de estudos pessoal, sem relação com os repositórios CETESB/GPLA da configuração global).

Revisão feita por um segundo agente (modelo Fable) sobre esta versão do plano: arquitetura aprovada na essência, com correções pontuais já incorporadas abaixo (marcadas onde relevante) — ver §7 para o resumo dos ajustes e o fatiamento de implementação recomendado.

---

## 1. O que a inspeção do código confirmou (fatos, não hipóteses)

### 1.1 Pipeline de modelos hoje (`refacer.py`)

| Papel | Classe/arquivo | Modelo | Onde carrega |
|---|---|---|---|
| Detecção | `SCRFD` (`recognition/scrfd.py`) | `det_10g.onnx` (buffalo_l) | `refacer.py:879-880` |
| Embedding/reconhecimento | `ArcFaceONNX` (`recognition/arcface_onnx.py`) | `w600k_r50.onnx` (buffalo_l) | `refacer.py:882-886` |
| Swap | `INSwapper` (pacote `insightface`, `insightface.model_zoo.inswapper`) | `inswapper_128.onnx` | `refacer.py:888-903` |

Não há `insightface.app.FaceAnalysis` em uso — o projeto usa somente os módulos vendorizados `SCRFD`/`ArcFaceONNX`, mais os tipos `Face`/`INSwapper`/`ensure_available` importados diretamente do pacote `insightface` (`refacer.py:16,19,20`).

### 1.2 Contrato de "rosto de destino preparado" hoje

`prepare_faces` (`refacer.py:1370-1406`) transforma cada entrada da UI em uma tupla:

```
(feat_original: np.ndarray | None, dest_face: insightface.app.common.Face, threshold: float)
```

- `dest_face` é criado em `__get_faces` (`refacer.py:1408-1424`): `Face(bbox=bbox, kps=kps, det_score=det_score)` com `.embedding` setado manualmente a partir de `self.rec_app.get(frame, kps)`.
- `_apply_swaps` (`refacer.py:1141-1264`) consome essa tupla: `feat_original` só é usado em `self.rec_app.compute_sim(rep_face[0], face.embedding)` (linha 1222); `dest_face` (índice 1) é o objeto passado para `self.face_swapper.get(frame, face, dest_face, paste_back=True)` (`_swap_one`, `refacer.py:1134-1139`) — o `INSwapper` consome `dest_face.embedding` internamente.
- **Confirmado via Context7** (`deepinsight/insightface`, `examples/in_swapper/README.md`): *"It is important to note that the INSwapper currently only accepts latent embeddings from the `buffalo_l` arcface model. Using embeddings from other models may result in abnormal output."* — ou seja, qualquer identidade agregada **tem que** produzir um `Face.embedding` no mesmo espaço vetorial do `w600k_r50.onnx` já usado. Isso descarta de saída qualquer ideia de reextrair embeddings com outro modelo (ex.: `FaceAnalysis` com outro pacote de modelos, ArcFace de outra dimensão, etc.).

### 1.3 UI hoje (`app.py`)

- Cada `gr.Tab("Face #N")` em Video Mode (`app.py:628-637`) tem `origin` (imagem única, "Face to replace") e `destination = gr.Gallery(..., file_types=["image"])` (`app.py:632`) — já aceita múltiplas imagens por slot, mas `_build_video_face_jobs` (`app.py:233-287`) trata cada imagem da galeria como **um job de reface independente**, nunca como fusão de identidade. Não há agregação de embeddings hoje em lugar nenhum do código.
- **Não existe nenhum `gr.State` no app hoje** — confirmado por busca no arquivo inteiro. O único estado global é a variável de módulo `video_history` e a instância única `refacer` compartilhada entre requisições (`app.py:26,128`).
- **Confirmado via Context7** (`gradio-app/gradio`, guia `state-in-blocks.md`): `gr.State` precisa ser deep-copy-ável; guarda dado por sessão (não compartilhado entre usuários); some ao fechar a aba após ~60 min (configurável via `delete_cache` em `gr.Blocks`). É o mecanismo correto para guardar o "perfil de identidade" em memória durante a sessão, sem introduzir infra nova.
- `gr.Progress(track_tqdm=True)` já é usado em `run()` (`app.py:289`) — mesmo padrão pode ser reaproveitado para a barra de progresso da extração de identidade.

### 1.4 Sem cache/persistência de identidade hoje

As únicas serializações em disco (`.npz`, `.json`) pertencem ao cache de frames de **vídeo-alvo** (`analyze_target_video`/`reface_with_cache`), não a perfis de identidade. Não há pickle nem qualquer formato de "perfil exportável" hoje — será uma peça nova, não uma extensão de algo existente.

---

## 2. Decisão de arquitetura (compatível com o pipeline existente)

**Reusar 100% do pipeline atual de detecção+embedding, sem introduzir modelo novo.** A "extração de identidade" é uma aba nova que:

1. Recebe N imagens e/ou M vídeos.
2. Para cada frame relevante, roda exatamente `self.face_detector.detect(...)` + `self.rec_app.get(frame, kps)` — as mesmas chamadas já usadas em `__get_faces`/`analyze_video_in_memory`. Nenhum modelo novo, nenhum download novo.
3. Agrupa (clusteriza) os embeddings resultantes por similaridade de cosseno usando `self.rec_app.compute_sim` — a mesma função já usada em `_apply_swaps`. Não introduz `sklearn`/`scipy`/HDBSCAN — ver §4.2 para a justificativa de usar clustering greedy simples em vez de dependência nova.
4. Para cada cluster ("Pessoa 1", "Pessoa 2", ...), filtra outliers de qualidade (desfoque, tamanho, pose) e calcula um **embedding médio normalizado** (centróide L2-normalizado) — esse é o "perfil".
5. O perfil resultante é embrulhado num `Face` sintético (`bbox`/`kps` de uma amostra representativa "melhor" do cluster, `.embedding` = centróide) — **exatamente o mesmo tipo de objeto que `dest_face` já é hoje**. Isso significa que `_apply_swaps`/`INSwapper.get` não precisam de nenhuma mudança de contrato: um perfil agregado é, no fim, um `Face` com embedding — indistinguível de um `dest_face` extraído de uma foto única.
   - **Confirmação adicional (revisão Fable)**: o `INSwapper.get()` consome apenas `source_face.normed_embedding` (derivado de `.embedding`) e ignora bbox/kps do rosto de origem. Ou seja, o `bbox`/`kps` do `Face` sintético é puramente cosmético (serve só para exibir a amostra representativa na UI), não afeta o swap em si — reforça que o centróide é suficiente e correto.

Consequência direta: **o modo atual de foto única continua existindo e funcionando sem tocar em `_apply_swaps`**. A nova aba apenas oferece uma *forma alternativa* de produzir o mesmo tipo de objeto (`Face` com `.embedding`) que hoje só vem de uma imagem, e esse objeto entra no Video Mode pelo mesmo canal que já existe (a Galeria de destino, ou uma extensão dela — ver §3.3). **Correção em relação a uma versão anterior deste plano**: `prepare_faces` recebe sim uma mudança pequena e localizada (um branch novo para aceitar a chave `identity_profile`, ver §3.3) — não é "zero alteração", é uma alteração aditiva e de baixo risco. `_apply_swaps` de fato não muda, pois consome sempre o mesmo tipo `Face` independentemente da origem.

---

## 3. Fluxo da nova aba "Criar Identidade" (somente pipeline de vídeo consome o resultado)

### 3.1 Sub-etapas da aba (conforme pedido: preparação → extração → revisão → teste → exportação → limpeza)

1. **Preparação / upload**
   - `gr.File(file_count="multiple", file_types=["image", "video"])` ou dois componentes separados (`gr.Gallery` para imagens + `gr.File(file_count="multiple")` para vídeos) — decisão de UX menor, não estrutural.
   - Checkbox obrigatório: *"Confirmo que possuo autorização para processar este material e estou ciente de que embeddings faciais são dados biométricos sensíveis."* — sem marcar, o botão de extração fica desabilitado (`interactive=False` até o checkbox mudar).
   - Aviso textual fixo (Markdown) sobre processamento local, nenhuma transmissão externa.

2. **Extração**
   - Reaproveita `self.face_detector.detect` (para vídeo, mesmo padrão de `analyze_video_in_memory`: decodifica com passo `skip_rate` para não processar todos os frames de vídeos longos) e `self.rec_app.get` para embedding, por candidato de rosto.
   - Progresso via `gr.Progress(track_tqdm=True)`, igual ao já usado em `run()`.
   - Filtro de qualidade **antes** de entrar no clustering (descarta, não silencia — expõe contagem ao usuário):
     - `det_score` abaixo de limiar (já disponível em `Face.det_score`, nenhum cálculo novo).
     - bbox pequena demais (rosto distante) — razão simples área-bbox/área-frame.
     - nitidez baixa (variância do Laplaciano no crop alinhado — operação barata em `cv2`, já é dependência do projeto, não introduz lib nova).
     - pose extrema — heurística barata a partir dos 5 keypoints (assimetria olho-nariz-boca), reaproveitando o que a limitação #19 do `REVIEW_GERAL_VIDEO.md` já documentou sobre o alinhamento 2D do projeto.

3. **Clustering em "Pessoa 1", "Pessoa 2", ...**
   - Greedy por similaridade de cosseno contra o centróide de cada cluster existente (usar `self.rec_app.compute_sim`, sem lib de clustering nova — ver §4.2).
   - **Correção (revisão Fable)**: atribuir cada amostra ao cluster de **maior** similaridade dentre os que passam o threshold, não ao primeiro cluster encontrado acima do threshold. A versão "primeiro acima do threshold" é sensível à ordem de processamento e deixa o centróide sofrer drift a partir de uma amostra ruim early — o custo de calcular o argmax contra todos os centróides existentes é desprezível (poucos clusters, poucos embeddings) e elimina essa fragilidade.
   - Nomes sempre neutros (`Pessoa {n}`), nunca inferidos de metadado de arquivo/nome.
   - Deixar explícito na implementação: o centróide é a média dos embeddings **individualmente L2-normalizados antes de somar**, com o resultado renormalizado no final — se a soma for feita sobre embeddings não normalizados, amostras com norma maior (ex. iluminação/captura diferente) dominam indevidamente o centróide.

4. **Revisão**
   - `gr.Gallery` por cluster mostrando as faces alinhadas (thumbnails), com botão para o usuário mover/remover amostras manualmemte antes de fechar o perfil (correção humana do clustering automático).
   - Contagem de amostras válidas vs. descartadas exibida.
   - **Nota de fatiamento (revisão Fable)**: mover itens entre galerias com estado por item é a parte mais cara de implementar em Gradio (eventos de seleção em `gr.Gallery` são limitados). Para a primeira versão, escopar somente "remover amostra" (não "mover entre clusters") — cobre a maior parte do valor prático e evita o componente mais caro da UI. Ver §7 para o fatiamento completo.

5. **Teste**
   - Pequeno preview: aplicar o perfil escolhido num frame de exemplo (do próprio material de origem ou de uma imagem de teste), reaproveitando `_swap_one`, para o usuário validar visualmente antes de "fechar" o perfil.
   - **Correção (revisão Fable)**: `_swap_one` espera um `Face` já detectado no frame-alvo (não roda detecção sozinho) — o preview precisa primeiro rodar `self.face_detector.detect`/`autodetect` sobre o frame de teste para obter esse `Face`, e só então chamar `_swap_one(frame, face_detectado, perfil_sintetico)`. Passo trivial, mas precisa constar explicitamente na implementação.

6. **Exportação / Limpeza**
   - Botão exportar perfil (ver §3.4) com aviso pré-exportação.
   - Botão "apagar identidade" (limpa o `gr.State` correspondente).
   - Botão "apagar temporários" (remove frames/crops intermediários gravados em `tmp/`, se o volume justificar não manter tudo em RAM — ver §4.3 sobre limite de memória).

### 3.2 Armazenamento durante a sessão

Perfis vivem em um `gr.State` (dict `{nome_cluster: {"embedding": np.ndarray, "thumbnail": np.ndarray, "n_samples": int, "quality_stats": {...}}}`), passado como input/output explícito nos event handlers relevantes — mesmo padrão documentado no guia oficial de `state-in-blocks.md` do Gradio. Nenhuma instância global nova: a única variável de módulo compartilhada continua sendo o objeto `refacer` (modelos), nunca os dados de identidade (que são por sessão).

### 3.3 Como o Video Mode consome o perfil (sem quebrar o fluxo de foto única)

A "Destination face(s)" do Video Mode já é uma `gr.Gallery` (`app.py:632`) tratada por imagem individual em `_build_video_face_jobs`. Duas opções, ambas aditivas:

- **Opção A (mínima, recomendada)**: adicionar um `gr.Dropdown` opcional por Face-slot, populado com os nomes dos perfis salvos no `gr.State` da aba de identidade ("nenhum" por padrão). Se um perfil for selecionado, `prepare_faces` recebe diretamente o `Face` sintético do perfil no lugar de rodar `__get_faces` sobre uma imagem. **Correção (revisão Fable)**: isso exige sim uma mudança pequena e localizada em `prepare_faces` — hoje o corpo chama `self.__get_faces(face['destination'])` incondicionalmente (`refacer.py:1387`); é preciso um branch que, ao encontrar a chave `identity_profile` no dict, use o `Face` sintético diretamente em vez de rodar detecção. A *assinatura* da função não muda (mesmos parâmetros), só o corpo ganha um `if`/`else` novo — grau de risco baixo, mas não é "zero alteração" como uma versão anterior deste plano chegou a sugerir no §6.
- **Opção B (não recomendada agora)**: fazer o perfil "virar" uma imagem sintética (ex. renderizar o crop alinhado como thumbnail e tratá-lo como se fosse upload) para reusar o caminho de `destination` sem tocar em `prepare_faces`. Mais simples de implementar, mas descarta a robustez do embedding agregado (thumbnail é só 1 amostra) — contraria o objetivo da funcionalidade. **Rejeitada.**

A Opção A é a única que preserva o ganho real (embedding centróide de várias amostras) sem exigir migração do fluxo de foto única — que continua batendo em `destination` (imagem) exatamente como hoje.

**Pontos de plumbing adicionais identificados na revisão (Fable) e não cobertos na versão anterior deste plano:**

- **`app.py:289-298` (`run()`) desempacota `*vars` por fatiamento posicional** (ex. `vars[1:num_faces+1]`, índices negativos como `vars[-5]`) para reconstruir os valores de cada componente Gradio. Adicionar um `gr.Dropdown` novo por Face-slot muda a lista de componentes de entrada do evento e, portanto, esse fatiamento posicional — é o ponto de maior risco de bug silencioso (índice errado, não erro de exceção) de toda a implementação. Precisa de atenção e teste manual específico, não é um acréscimo "livre de efeito colateral".
- **`gr.Dropdown` não atualiza sozinho a lista de `choices` quando o `gr.State` de perfis muda.** É preciso um evento explícito — por exemplo, o handler que fecha/salva um perfil na aba de identidade deve retornar `gr.update(choices=[...])` para cada dropdown de Face-slot no Video Mode (ou popular os choices no evento `select` da aba Video Mode). Sem isso, o dropdown fica com a lista vazia/desatualizada mesmo depois de o perfil existir no estado.
- **Volatilidade de sessão no Colab**: `gr.State` é perdido em reload de página ou reconexão — e o link público do Gradio usado em Colab reconecta com alguma frequência (queda de rede, timeout do túnel). Isso eleva a exportação `.npz` (§3.4) de "feature opcional de fim de fluxo" para **mitigação de perda de trabalho** — vale sugerir ao usuário exportar logo após fechar um perfil, não só no final de toda a sessão.
- **Modos Multiple Faces e Faces By Match**: o plano original não definia se o dropdown de perfil deveria valer nesses modos. Decisão proposta (a confirmar com o usuário, ver §4.4): em "Multiple Faces" o `origin` já é ignorado e o matching é só por posição — o dropdown de perfil deveria valer normalmente ali (só troca a fonte do `dest_face`, não mexe na lógica posicional). Em "Faces By Match" o dropdown também deveria valer (troca a fonte de `feat_original`/`dest_face` normalmente). Ou seja: o dropdown é ortogonal aos 3 modos de matching — ele decide **de onde vem o rosto de destino**, não como o matching é feito. Precisa só de teste explícito nos 3 modos antes de considerar concluído.

### 3.4 Exportação / importação de perfil

- Formato: `.npz` (mesma lib já usada no cache de vídeo, nenhuma dependência nova) contendo `embedding` (vetor), `metadata` mínima (nome do cluster, contagem de amostras, versão do modelo de embedding usado — `w600k_r50`/buffalo_l, timestamp), e opcionalmente o thumbnail (imagem pequena, já borrada/reduzida) para conferência visual ao reimportar.
- **Não** incluir os embeddings de todas as amostras individuais no export — só o centróide final, reduzindo a superfície de dado biométrico exposto no arquivo.
- Aviso obrigatório antes de exportar: *"Este arquivo contém dados biométricos derivados de rosto(s). Trate como informação sensível. Não compartilhe sem autorização."*
- Import: valida a versão do modelo de embedding registrada no metadata contra `w600k_r50`/dimensão atual antes de aceitar — rejeita misturar espaços vetoriais incompatíveis (mitigação direta ao risco de "perfis adulterados" pedido no escopo). **Correção (revisão Fable)**: essa rejeição deve ser **ruidosa** (`gr.Error` explícito com a razão), não silenciosa — um perfil rejeitado sem feedback deixa o usuário sem entender por que "nada aconteceu" ao importar. Validação de formato: usar `np.load(..., allow_pickle=False)` (evita execução arbitrária de código via pickle malicioso — mitigação a "formatos de arquivo maliciosos").

---

## 4. Riscos, limitações e decisões que dependem de validação com o usuário

### 4.1 Clustering: greedy incremental, não algoritmo novo

Não introduzir `scikit-learn`/`hdbscan`/`scipy.cluster` — o projeto não tem nenhuma dependência de ML clássico hoje (só ONNX Runtime + OpenCV + numpy). Um clustering greedy incremental (compara contra centróides existentes, cria cluster novo se similaridade abaixo do threshold de todos) usa só `numpy` (já presente) e a função `compute_sim` já existente. Evita dependência nova e evita risco de conflito de versão do NumPy 1.24.3 fixado no `requirements-GPU.txt` (bibliotecas de clustering mais novas frequentemente pedem NumPy 2.x).

### 4.2 Extração de frames de vídeo para identidade: reaproveitar, não duplicar, a lógica de decode

`analyze_video_in_memory` já decodifica com controle de `skip_rate` e teto de RAM (`in_memory_analysis_max_mb`, `refacer.py:94-100`). A extração de identidade a partir de vídeo deve reaproveitar esse padrão de decode com passo (não decodificar todo frame de vídeos longos) — evita adicionar um segundo caminho de decodificação de vídeo ao projeto.

### 4.3 Armazenamento temporário em Colab

Dado "armazenamento temporário" como restrição confirmada: preferir manter crops/embeddings em RAM durante a sessão de extração (mesmo padrão do `analyze_video_in_memory`), gravando em `tmp/` (diretório já existente no projeto, `tmp/`) apenas se o volume de material subir além de um teto configurável — mesmo padrão de fallback já usado no cache de vídeo. Botão de limpeza remove esse diretório.

### 4.4 O que precisa de confirmação do usuário antes de implementar

- **Limiar de similaridade padrão para separar clusters distintos** ("Pessoa 1" vs "Pessoa 2"). **Correção (revisão Fable)**: o plano original sugeria emprestar o 0.2 do slider de Faces By Match como ponto de partida — isso está errado para este uso. O 0.2 é deliberadamente permissivo porque ali a similaridade serve para *confirmar* uma identidade já conhecida dentro do mesmo vídeo; para *separar pessoas distintas* em material arbitrário (o objetivo do clustering aqui), um threshold baixo funde pessoas parecidas no mesmo cluster. Partir de ~0.30–0.35 (faixa usual de separação inter-pessoa para embeddings ArcFace) e validar/ajustar visualmente na etapa de Revisão.
- Se a Opção A (dropdown de perfil por Face-slot) deve coexistir com a Galeria de destino existente ou substituí-la quando um perfil é selecionado — **decisão recomendada pela revisão (Fable)**: desabilitar a Galeria daquele Face-slot quando um perfil estiver selecionado no dropdown. Ambiguidade sobre qual fonte de destino vale é pior do que uma UI temporariamente travada; ir direto para essa opção em vez de tratar como aberta.
- Limite de imagens/vídeos de entrada por sessão de extração (relevante para RAM/tempo, mas é escolha de produto, não técnica).

---

## 5. Registro de consultas ao Context7

| Biblioteca | ID Context7 | Assunto consultado | Conclusão | Impacto na arquitetura |
|---|---|---|---|---|
| Gradio | `/gradio-app/gradio` | `gr.State` (persistência por sessão), `gr.Progress(track_tqdm=True)` | Confirmado: `gr.State` guarda dado por sessão, precisa ser deep-copy-ável, dado com timeout configurável (`delete_cache`); progress via tqdm já é o padrão usado no projeto | Usar `gr.State` para o dict de perfis é seguro e é o mecanismo oficial recomendado — nenhuma dependência nova |
| InsightFace | `/deepinsight/insightface` | Contrato do `INSwapper.get()` quanto à origem do embedding aceito | Confirmado (doc oficial do exemplo `in_swapper`): **INSwapper só aceita embeddings do modelo arcface `buffalo_l`** — outros modelos produzem saída anormal | Restrição dura: a identidade agregada deve ser produzida exclusivamente com `ArcFaceONNX`/`w600k_r50.onnx` já em uso — descarta qualquer `FaceAnalysis`/modelo alternativo para o embedding do perfil |

Nenhuma outra biblioteca foi consultada por não haver lacuna identificada que justificasse (não há necessidade de consultar `numpy`/`opencv` — uso já estabelecido e trivial no projeto).

---

## 6. Resumo do que NÃO muda

- `_apply_swaps`, `reface`, todo o pipeline de Image/GIF/TIFF Mode: **inalterados**.
- `prepare_faces` recebe uma alteração pequena e aditiva (branch novo para a chave `identity_profile`, ver §3.3) — **não** é "inalterado" como uma versão anterior deste plano afirmava; é baixo risco, mas real.
- Nenhum modelo novo baixado; nenhuma dependência nova instalada.
- O fluxo de foto única em vídeo continua idêntico quando nenhum perfil é selecionado no dropdown novo.
- Nenhuma transmissão externa de embeddings/imagens; processamento 100% local na sessão do Colab.

---

## 7. Revisão cruzada (modelo Fable) e fatiamento de implementação recomendado

Este plano foi revisado por um segundo agente (Fable) atuando como arquiteto sênior consultado informalmente, com leitura direta do código para conferir as citações. Veredito: **arquitetura aprovada na essência** — reuso do ArcFace existente, `Face` sintético com centróide, `gr.State` para sessão e clustering greedy sem lib nova estão todos corretos e já incorporados nas seções acima. As correções pontuais (contradição sobre `prepare_faces`, plumbing de `app.py`, threshold de clustering, greedy por argmax, UX de erro no import, etc.) já foram aplicadas nas seções correspondentes.

O ponto de maior valor da revisão foi o **fatiamento de implementação**, que inverte a ordem original do plano (que descrevia o fluxo completo upload→extração→clustering→revisão→teste→export como um bloco único). Ordem recomendada:

1. **Fatia 1 — validar a hipótese central primeiro**: extração multi-imagem assumindo uma única pessoa no material (sem clustering multi-pessoa), filtro de qualidade, centróide, `Face` sintético, e o consumo no Video Mode (dropdown + branch em `prepare_faces` + ajuste do fatiamento posicional em `app.py:run()`). Isso valida de ponta a ponta, com o mínimo de UI, que o centróide funciona corretamente dentro do `INSwapper` — a parte de maior incerteza técnica real.
2. **Fatia 2 — export/import `.npz`**: pequena, e resolve cedo o risco de perda de perfil por volatilidade de sessão no Colab (§3.3).
3. **Fatia 3 — vídeos como fonte de extração**: reuso do padrão de decode com `skip_rate` já existente em `analyze_video_in_memory`.
4. **Fatia 4 — clustering multi-pessoa + UI de revisão** (mover/remover amostras): a parte mais cara de UI, e dispensável para o caso de uso dominante ("material de uma única pessoa"), onde o filtro de qualidade + corte de similaridade contra o centróide já bastam como filtro de outlier.
5. **Fatia 5 — preview/teste do perfil** antes de fechar/exportar.

Racional: o clustering multi-pessoa estava posicionado no plano original como núcleo da funcionalidade, mas é na prática a peça mais adiável — o caso de uso mais comum (várias fotos/vídeos da mesma pessoa) não depende dele para gerar valor.
