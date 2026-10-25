import hashlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from src.assets import download_archive


class DownloadTests(TestCase):
    def test_interrupted_download_resumes_and_verifies_bytes(self):
        payload = b"pinned dataset for the test"
        cfg = dict(data_url="https://example.org/data",data_bytes=len(payload),
                   data_sha256=hashlib.sha256(payload).hexdigest())
        with TemporaryDirectory() as folder:
            cache = Path(folder)
            (cache/"hpatches.zip.part").write_bytes(payload[:7])
            def response(request, timeout):
                self.assertEqual(request.get_header("Range"),"bytes=7-")
                remaining = io.BytesIO(payload[7:])
                remaining.status = 206
                remaining.headers = {"Content-Range":f"bytes 7-{len(payload)-1}/{len(payload)}"}
                return remaining
            with patch("src.assets.urlopen", side_effect=response):
                download_archive(cache,cfg,cache/"complete.zip")
            self.assertEqual((cache/"complete.zip").read_bytes(),payload)
            self.assertFalse((cache/"hpatches.zip.part").exists())
