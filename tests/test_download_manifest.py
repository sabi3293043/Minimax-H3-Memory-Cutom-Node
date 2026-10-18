import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("download_models", Path(__file__).resolve().parents[1] / "download_models.py")
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


class DownloadTests(unittest.TestCase):
    def test_integrity_check_rejects_truncated_and_modified_files(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture"
            path.write_bytes(b"valid artifact")
            digest = hashlib.sha256(b"valid artifact").hexdigest()
            download.verify(path, 14, digest)
            with self.assertRaisesRegex(ValueError, "file size"):
                download.verify(path, 15, digest)
            path.write_bytes(b"VALID artifact")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                download.verify(path, 14, digest)


if __name__ == "__main__":
    unittest.main()
