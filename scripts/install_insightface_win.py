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
from pathlib import Path

VERSION = "0.7.3"  # mesma versão usada no Colab
BASE_URL = "https://github.com/Gourieff/Assets/raw/main/Insightface"
SUPPORTED = {(3, 10), (3, 11), (3, 12)}

# insightface/app/__init__.py importa mask_renderer, que importa thirdparty.face3d,
# cujo mesh_core_cython.pyd na wheel pré-compilada foi ligado contra NumPy 1.x.
# Com NumPy 2 esse import morre em "numpy.dtype size changed" e derruba qualquer
# uso do pacote — inclusive app.common.Face, que o projeto realmente usa.
# Rebaixar o NumPy não é opção: opencv-python 4.13 e imagecodecs exigem >=2.
# Nada aqui referencia MaskRenderer nem face3d, então o módulo é substituído por
# um stub que preserva o nome e só falha se alguém tentar instanciá-lo.
MASK_RENDERER_STUB = '''"""Substituído por scripts/install_insightface_win.py.

O módulo original importa thirdparty.face3d, cuja extensão compilada exige
NumPy 1.x e é incompatível com o NumPy 2 usado por este projeto. Nenhum
código deste repositório usa MaskRenderer.
"""


class MaskRenderer:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "MaskRenderer foi desativado nesta instalação: depende de "
            "insightface.thirdparty.face3d, incompatível com NumPy 2."
        )
'''


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


def patch_mask_renderer(site_packages=None):
    """Substitui insightface/app/mask_renderer.py pelo stub. Idempotente."""
    if site_packages is None:
        import insightface

        destino = Path(insightface.__file__).parent / "app" / "mask_renderer.py"
    else:
        destino = Path(site_packages) / "insightface" / "app" / "mask_renderer.py"

    if not destino.exists():
        print(f"[AVISO] {destino} não encontrado; nada a corrigir.")
        return False

    if destino.read_text(encoding="utf-8").startswith('"""Substituído por'):
        print("mask_renderer já corrigido.")
        return True

    destino.write_text(MASK_RENDERER_STUB, encoding="utf-8")
    print(f"mask_renderer neutralizado em: {destino}")
    return True


def main():
    url = wheel_url()
    print(f"Instalando insightface {VERSION} de: {url}")
    resultado = subprocess.run(
        [sys.executable, "-m", "pip", "install", url],
        check=False,
    )
    if resultado.returncode != 0:
        return resultado.returncode

    patch_mask_renderer()
    return 0


if __name__ == "__main__":
    sys.exit(main())
