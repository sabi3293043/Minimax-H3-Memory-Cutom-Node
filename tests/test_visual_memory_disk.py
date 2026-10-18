"""Disk session lifecycle tests: no ComfyUI or accelerator required."""
import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mosaichunk_h3.visual_memory_disk import VisualMemoryDisk


class VisualMemoryDiskTests(unittest.TestCase):
    def test_save_read_and_complete(self):
        with tempfile.TemporaryDirectory() as root:
            with VisualMemoryDisk(root, max_bytes=24) as bank:
                record = bank.save(step=1, frame=12, data=b"encoded-data")
                self.assertEqual(bank.read(record), b"encoded-data")
                run = bank.directory
                self.assertTrue(run.exists())
                self.assertEqual(VisualMemoryDisk.cleanup_abandoned(root), 0)
            self.assertFalse(run.exists())

    def test_cleanup_on_failure_or_interrupt(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                with VisualMemoryDisk(root) as bank:
                    run = bank.directory
                    bank.save(step=0, frame=0, data=b"image")
                    raise RuntimeError("interrupted")
            self.assertFalse(run.exists())

    def test_budget_and_closed_session(self):
        with tempfile.TemporaryDirectory() as root:
            bank = VisualMemoryDisk(root, max_bytes=5)
            bank.save(step=0, frame=0, data=b"abc")
            with self.assertRaises(MemoryError):
                bank.save(step=0, frame=1, data=b"xyz")
            self.assertEqual(bank.used_bytes, 3)
            bank.close()
            bank.close()
            with self.assertRaises(RuntimeError):
                bank.save(step=0, frame=2, data=b"a")

    def test_orphan_recovery_skips_unrelated_files(self):
        with tempfile.TemporaryDirectory() as root:
            with VisualMemoryDisk(root) as bank:
                orphan = bank.directory
            orphan.mkdir()
            (orphan / ".lock").write_bytes(b"0")
            other = Path(root) / "not-a-session"
            other.mkdir()
            (other / "keep").write_text("safe")
            self.assertEqual(VisualMemoryDisk.cleanup_abandoned(root), 1)
            self.assertFalse(orphan.exists())
            self.assertTrue((other / "keep").exists())


if __name__ == "__main__":
    unittest.main()
