import importlib.util
from pathlib import Path
import subprocess
import zipfile


ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("exp023_packager",ROOT/"scripts/package_exp023.py")
packager=importlib.util.module_from_spec(spec)
spec.loader.exec_module(packager)


def test_archive_preserves_git_bytes_with_windows_autocrlf(tmp_path):
    repository=tmp_path/"repository"
    repository.mkdir()
    def git(*args):
        return subprocess.check_output(["git",*args],cwd=repository,stderr=subprocess.DEVNULL)
    git("init")
    git("config","core.autocrlf","true")
    (repository/"config.json").write_bytes(b'{\n  "fixed": true\n}\n')
    git("add","config.json")
    git("-c","user.name=CPU fixture","-c","user.email=fixture@example.invalid","commit","-m","fixture")
    expected=git("show","HEAD:config.json")
    destination=tmp_path/"source.zip"
    packager.archive_source(repository,"HEAD",destination)
    with zipfile.ZipFile(destination) as z:
        assert z.read("config.json")==expected
        assert b'\r\n' not in z.read("config.json")
