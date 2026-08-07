"""Execução de sessões ONNX resiliente à codificação das mensagens de erro.

O onnxruntime decodifica as mensagens vindas do runtime nativo como UTF-8. Em
Windows não-inglês o DirectML devolve texto na codificação local (cp1252 em
português), e qualquer acento faz a decodificação estourar um UnicodeDecodeError
que substitui o erro original — a falha real do provider se perde por completo,
inclusive nos logs.

run_session() executa a sessão e, quando isso acontece, recupera a mensagem
bruta para que a causa apareça no lugar do erro de codificação.

Também serializa as chamadas: as sessões do detector e do reconhecedor são
compartilhadas entre o pipeline de vídeo e a construção de perfis de
identidade (identity_profile recebe as instâncias de refacer), e o
onnxruntime não garante execução concorrente segura no DirectML. Sem isso,
usar as duas telas ao mesmo tempo corrompe a sessão, que passa a falhar até o
aplicativo ser reiniciado.
"""

import threading

# Um lock por objeto de sessão: chamadas em sessões distintas seguem paralelas,
# e apenas as que disputam a mesma sessão são serializadas.
_locks = {}
_locks_guard = threading.Lock()


def _lock_da_sessao(session):
    chave = id(session)
    lock = _locks.get(chave)
    if lock is None:
        with _locks_guard:
            lock = _locks.get(chave)
            if lock is None:
                lock = threading.Lock()
                _locks[chave] = lock
    return lock


def run_session(session, output_names, input_feed, run_options=None):
    """Executa session.run preservando a mensagem de erro original."""
    try:
        with _lock_da_sessao(session):
            return session.run(output_names, input_feed, run_options)
    except UnicodeDecodeError as erro:
        bruto = getattr(erro, "object", b"")
        if isinstance(bruto, (bytes, bytearray)):
            # errors="replace" mantém o trecho legível: o que interessa é a
            # descrição do provider, não os caracteres acentuados em si.
            mensagem = bytes(bruto).decode("utf-8", errors="replace").strip()
        else:
            mensagem = str(bruto)
        raise RuntimeError(
            "Falha na execução da sessão ONNX. A mensagem original do provider "
            "chegou em codificação não-UTF-8 e foi recuperada abaixo:\n"
            f"{mensagem}"
        ) from erro
