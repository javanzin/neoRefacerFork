# Plano Técnico — Identity Evolutivo (versionamento incremental de perfis)

Status: **implementado** (formato v2, fluxo de continuação incremental e UI de import/continuar/comparar/confirmar). Passou por code review adversarial (achou e corrigiu uma RCE via `allow_pickle=True`, viés de peso em `merge_imported_profile`, gate de consentimento, acumulação de candidata entre rodadas — ver histórico do módulo). Ainda **não validado em ambiente real** (Colab/Lightning.ai com dependências completas) — validado até aqui só via testes unitários de `identity_profile.py`.
**Requisito não negociável**: perfis `.npz` já exportados no formato atual devem continuar funcionando exatamente como hoje no Video Mode (upload direto no Face-slot, import via `import_profile`, swap via `prepare_faces`/`INSwapper`) — o novo formato é aditivo, nunca substitui nem invalida o antigo. Ver seção 5.
Projeto: NeoRefacer (app de estudos pessoal, sem relação com os repositórios CETESB/GPLA da configuração global).

---

## 1. A ideia

Hoje, refinar um Identity com material novo exige reenviar **todo** o material original (fotos + vídeos) e reextrair do zero — o `.npz` exportado guarda só o centroide final, sem nenhum dado por origem.

Ideia: permitir versionamento incremental —

```
Identity v1 = fotos e vídeos iniciais
Identity v2 = v1 + novas fotos
Identity v3 = v2 + novo vídeo
```

— importando a v1 já exportada, adicionando só a mídia nova, gerando uma v2 candidata, comparando com a v1 antes de decidir, e exportando a v2 como um novo arquivo (sem banco de dados, sem estado servidor — cada versão é um `.npz` autocontido, consistente com o resto do projeto).

## 2. Por que o formato atual não permite isso

`export_profile`/`import_profile` (`identity_profile.py`) gravam e leem apenas: `embedding`, `bbox`, `kps`, `det_score`, `name`, `n_samples`, `embedding_model`, `created_at`. Nenhum dado por origem, nenhuma amostra individual (design de privacidade deliberado, documentado no próprio módulo).

O swapper (`INSwapper.get()`, confirmado via Context7) só consome `.embedding`/`.normed_embedding` do `Face` — `bbox`/`kps`/`det_score` no perfil importado hoje só servem para exibir o "representative" (thumbnail/preview), não afetam o swap.

`n_samples` é só uma contagem informativa (exibição) — nunca é lido por nenhuma função de agregação (`_compute_centroid`, `_compute_balanced_centroid`, `merge_profiles`), então não pode servir de peso real hoje.

## 3. Novo formato de exportação (v2, versionado)

Precisa ser uma mudança de **formato**, não uma extensão do atual — com uma chave `profile_format_version` explícita no `.npz`, validada em `import_profile` com o mesmo padrão já usado para `embedding_model` (rejeição ruidosa, `gr.Error`/`ValueError` explícito, nunca silenciosa).

Por origem, gravar:
- `origin` (identificador cru — nome do arquivo/vídeo).
- `centroid` (vetor L2-normalizado daquela origem, float32, 512-d — mesmo que `_origin_centroids_as_pseudo_samples` já calcula hoje em memória).
- `n_samples` (contagem, informativa — não peso).
- `content_hash` (SHA-256 do arquivo, mesmo mecanismo de `_hash_file`/`_dedupe_files_by_content` já existente em `app.py`, usado hoje só dentro da sessão).

Mais os campos já existentes hoje (`embedding` final, `bbox`, `kps`, `det_score`, `name`, `embedding_model`, `created_at`) para compatibilidade de leitura/preview sem precisar recalcular nada.

**Nunca gravar embeddings de amostras individuais** — mantém o mesmo nível de exposição de dado biométrico que o design atual já escolheu deliberadamente (só 1 vetor por origem, não N vetores por frame).

### Estimativa de tamanho

Por origem: centroide (512 × float32 = 2048 bytes) + hash (32 bytes) + metadados (~100-150 bytes) ≈ 2,2 KB. Para 1.000 origens: **≈ 2,2 MB** sem compressão (`np.savez` puro — `savez_compressed` ajudaria pouco, embeddings float32 quase-aleatórios comprimem mal). Desprezível frente ao custo de reprocessar o material original.

## 4. Fluxo de atualização incremental

1. Importar a v_N já exportada (novo formato) — carrega a lista de origens (cada uma com seu centroide) em vez de só o centroide final.
2. Usuário envia só a mídia nova (fotos/vídeos adicionais da mesma pessoa).
3. **Dedup**: antes de processar, comparar o hash de cada arquivo novo contra os `content_hash` já salvos na v_N — arquivo idêntico é ignorado, evitando reprocessamento/duplicação. Ressalva: hash de arquivo (SHA-256) só detecta byte-a-byte idêntico — recompressão, recorte ou reencode da mesma mídia não é detectado (limitação aceita, não resolvida nesta fase).
4. **Validação de identidade**: reaproveitar a busca dirigida já existente (`find_match_in_frame`/`find_matches_in_video`, mesmo mecanismo de `TARGET_MATCH_SIMILARITY_THRESHOLD`) para confirmar que a mídia nova é da mesma pessoa antes de incorporar — não é peça nova, é reuso direto.
5. Processar só a mídia nova (detecção + embedding + filtros de qualidade, pipeline já existente) e calcular os centroides das novas origens.
6. **Origem legada**: a v_N importada entra na combinação como **1 origem a mais**, com peso igual às demais (não escalado por `n_samples`) — escalar pelo `n_samples` da v_N reintroduziria o viés de volume que o balanceamento por origem existe para evitar, já que não há garantia sobre a distribuição interna de uma versão antiga sem seus centroides de origem individuais preservados (caso raro: importar um `.npz` do formato v1 atual, sem centroides por origem — nesse caso só resta tratá-lo como legado de origem única).
7. Gerar a **candidata** (v_N+1) combinando origens antigas + novas, sem sobrescrever a v_N — toda função de agregação (`_build_profile_from_samples`, `apply_anchor_to_profile`) já é pura (retorna novo dict, nunca muta in-place), então basta manter os dois perfis na mesma `gr.State` da sessão.
8. **Comparação antes de confirmar**: reaproveitar `preview_identity_swap` (já existente) para comparar visualmente v_N vs. candidata no mesmo destino de teste, antes de decidir.
9. Confirmar (exportar a candidata como novo `.npz`) ou descartar (manter só a v_N já exportada) — sem necessidade de infraestrutura nova, já que "desfazer" é simplesmente não promover/exportar a candidata.

### Foto âncora

Continua armazenada **separada** dos centroides de origem (mesmo desenho já implementado: etapa externa, aplicada depois do centroide base pronto). Isso permite remover/reaplicar a âncora livremente após qualquer atualização incremental do Identity base, sem recalcular nada além da combinação final.

## 5. Compatibilidade com perfis existentes

Perfis exportados no formato atual (sem `profile_format_version`, sem centroides por origem) **não** suportam evolução incremental nativa — só o centroide final está disponível, então a única opção é tratá-los como "origem legada única" (item 6 do fluxo acima) ao importar. Não há migração automática possível sem reprocessar o material original.

## 6. Riscos e limitações conhecidas

- Mídia nova pode piorar o resultado (pose ruim, oclusão, baixa qualidade) tanto quanto material inicial — daí a importância do passo de comparação antes de confirmar (item 8 do fluxo).
- Hash de arquivo não detecta duplicata "semântica" (mesma mídia reencodada/recortada) — só duplicata byte-a-byte.
- Peso de origem legada sem distribuição conhecida é uma decisão de design (peso igual, não escalado), não uma solução perfeita.
- Nenhum obstáculo técnico estrutural identificado — a parte matematicamente difícil (agregação por origem sem dominância de volume) já está resolvida pelo balanceamento por origem existente; o trabalho restante é de formato de arquivo e fluxo de UI, não de algoritmo novo.

## 7. Recomendação

Vale implementar no futuro, mas **depois** de validar o balanceamento por origem atual em uso real (não em paralelo). Prioridade sugerida:
1. Validar `balance_by_origin` em casos reais (vídeo longo + poucas fotos, conforme já discutido).
2. Desenhar e implementar o formato v2 de exportação (centroides por origem + hash + versão de formato).
3. Implementar o fluxo de importação incremental + comparação + confirmação/descarte.
4. Foto âncora permanece com o mesmo desenho já existente (upload manual separado, nenhuma mudança necessária).
