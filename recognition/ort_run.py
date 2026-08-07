"""Execução de sessões ONNX resiliente à codificação das mensagens de erro.

O onnxruntime decodifica as mensagens vindas do runtime nativo como UTF-8. Em
Windows não-inglês o DirectML devolve texto na codificação local (cp1252 em
português), e qualquer acento faz a decodificação estourar um UnicodeDecodeError
que substitui o erro original — a falha real do provider se perde por completo,
inclusive nos logs.

run_session() executa a sessão e, quando isso acontece, recupera a mensagem
bruta para que a causa apareça no lugar do erro de codificação.
"""


def run_session(session, output_names, input_feed, run_options=None):
    """Executa session.run preservando a mensagem de erro original."""
    try:
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
