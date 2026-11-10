"""Create an isolated Transformers overlay over the pinned baseline runtime."""

import importlib.metadata
import json
import os
import subprocess
import sys
import sysconfig
import venv
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    baseline = root / ".venv"
    if Path(sys.prefix).resolve() != baseline or importlib.metadata.version("transformers") != "4.49.0":
        raise SystemExit("Run this script from the pinned baseline .venv.")
    config = json.loads((root / "configs" / "exp006.json").read_text(encoding="utf-8"))
    overlay = root / ".venv-siglip2"
    python = overlay / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        venv.EnvBuilder(with_pip=True).create(overlay)
    purelib = subprocess.check_output([python, "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True)
    site = Path(purelib.strip())
    (site / "baseline-dependencies.pth").write_text(sysconfig.get_path("purelib") + "\n", encoding="utf-8")
    subprocess.run([python, "-m", "pip", "install", "--no-deps", f"transformers=={config['transformers_version']}"],
                   check=True)
    subprocess.run([python, "-m", "pip", "check"], check=True)
    subprocess.run([python, "-c", "import sys,torch,transformers; from pathlib import Path; "
                    "assert transformers.__version__ == sys.argv[1]; "
                    "assert Path(transformers.__file__).is_relative_to(Path(sys.prefix)); "
                    "print(transformers.__version__, torch.__version__, torch.cuda.is_available())",
                    config["transformers_version"]], check=True)
    print(f"Extraction interpreter is {python}")


if __name__ == "__main__":
    main()
