import hashlib
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import NamedTemporaryFile, TemporaryDirectory
from urllib.request import urlopen


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch_inputs(cache, cfg, need_data=True):
    cache.mkdir(parents=True, exist_ok=True)
    source = cache / "xfeat"
    if not source.exists():
        with TemporaryDirectory(dir=cache) as tmp:
            checkout = Path(tmp) / "source"
            subprocess.run(["git", "init", "--quiet", str(checkout)], check=True)
            subprocess.run(["git", "-C", str(checkout), "fetch", "--depth=1",
                            "https://github.com/verlab/accelerated_features.git", cfg["xfeat_commit"]], check=True)
            subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", "FETCH_HEAD"], check=True)
            checkout.rename(source)
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != cfg["xfeat_commit"] or digest(source/"weights/xfeat.pt") != cfg["weights_sha256"]:
        raise ValueError("The cached XFeat revision or weights differ from configs/benchmark.json.")
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise ValueError("Cached XFeat tracked files have local modifications.")
    data = cache / "hpatches.zip"
    if need_data:
        if not data.is_file():
            with NamedTemporaryFile(dir=cache, delete=False, suffix=".download") as destination:
                tmp = Path(destination.name)
            try:
                with urlopen(cfg["data_url"], timeout=120) as source_stream, tmp.open("wb") as stream:
                    shutil.copyfileobj(source_stream, stream, 1024*1024)
                if digest(tmp) != cfg["data_sha256"]:
                    raise ValueError("HPatches archive checksum mismatch")
                os.replace(tmp, data)
            finally:
                tmp.unlink(missing_ok=True)
        elif digest(data) != cfg["data_sha256"]:
            raise ValueError(f"HPatches checksum mismatch at {data}")
    return source, data
