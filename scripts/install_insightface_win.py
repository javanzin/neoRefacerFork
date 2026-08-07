"""Instala o insightface no Windows a partir de uma wheel pré-compilada.

O insightface não publica wheel para Windows em nenhuma versão — o PyPI só
oferece o .tar.gz, cuja build compila a extensão Cython
``face3d/mesh/cython/mesh_core`` e exige o Visual C++ Build Tools. Esse módulo
não é referenciado em nenhum ponto deste projeto, então instalar o compilador
serviria apenas para produzir código morto.

Roda com o interpretador da venv de destino, de modo que a tag de ABI usada na
URL é sempre a do ambiente que receberá o pacote.
"""

import subprocess
import sys

VERSION = "0.7.3"  # mesma versão usada no Colab
BASE_URL = "https://github.com/Gourieff/Assets/raw/main/Insightface"
SUPPORTED = {(3, 10), (3, 11), (3, 12)}


def wheel_url(version_info=None):
    """Monta a URL da wheel correspondente ao interpretador informado."""
    info = version_info or sys.version_info
    key = (info[0], info[1])
    if key not in SUPPORTED:
        versoes = ", ".join(f"{a}.{b}" for a, b in sorted(SUPPORTED))
        raise SystemExit(
            f"[ERRO] Python {key[0]}.{key[1]} não tem wheel pré-compilada do "
            f"insightface. Versões suportadas: {versoes}."
        )
    tag = f"cp{key[0]}{key[1]}"
    return f"{BASE_URL}/insightface-{VERSION}-{tag}-{tag}-win_amd64.whl"


def main():
    url = wheel_url()
    print(f"Instalando insightface {VERSION} de: {url}")
    resultado = subprocess.run(
        [sys.executable, "-m", "pip", "install", url],
        check=False,
    )
    return resultado.returncode


if __name__ == "__main__":
    sys.exit(main())
