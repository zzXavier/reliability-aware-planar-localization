import hashlib
import os
from pathlib import Path
import shutil
import subprocess
from filelock import FileLock
from tempfile import TemporaryDirectory
from http.client import IncompleteRead
from urllib.error import URLError
from urllib.request import Request, urlopen


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download_archive(cache, cfg, destination):
    # A reader may lose its connection after a gigabyte. Retain verified-length
    # progress per attempt, but expose only hash-verified complete archives.
    expected_size = cfg["data_bytes"]
    part = cache / "hpatches.zip.part"
    for attempt in range(3):
        offset = part.stat().st_size if part.exists() else 0
        if offset == expected_size and digest(part)==cfg["data_sha256"]:
            os.replace(part,destination)
            return
        if offset>=expected_size:
            part.unlink()
            offset=0
        request = Request(cfg["data_url"],headers={"Range":f"bytes={offset}-"} if offset else {})
        try:
            with urlopen(request,timeout=120) as response:
                append = bool(offset and response.status==206)
                if append and not response.headers.get("Content-Range","").startswith(f"bytes {offset}-"):
                    raise ValueError("Server returned a different byte range")
                with part.open("ab" if append else "wb") as stream:
                    shutil.copyfileobj(response,stream,1024*1024)
        except (URLError, OSError, IncompleteRead):
            if attempt == 2:
                raise
            continue
        if part.stat().st_size == expected_size and digest(part)==cfg["data_sha256"]:
            os.replace(part,destination)
            return
    raise ValueError(f"Incomplete or corrupt HPatches file at {part}; retry to resume")


def fetch_inputs(cache, cfg, need_data=True):
    cache.mkdir(parents=True, exist_ok=True)
    source = cache / "xfeat"
    with FileLock(str(cache/"xfeat.lock"), timeout=600):
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
            with FileLock(str(cache/"hpatches.lock"), timeout=2400):
                if not data.exists():
                    download_archive(cache,cfg,data)
                elif digest(data) != cfg["data_sha256"]:
                    raise ValueError(f"HPatches checksum mismatch at {data}")
        elif digest(data) != cfg["data_sha256"]:
            raise ValueError(f"HPatches checksum mismatch at {data}")
    return source, data
