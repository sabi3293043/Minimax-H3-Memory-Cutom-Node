"""The optional path must not install at import, upgrade torch or leak locks."""
import importlib
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from mosaichunk_h3 import optional_occlusion
from mosaichunk_h3.memory_controls import MemoryOptions


class OptionalOcclusionTests(unittest.TestCase):
    def test_import_and_disabled_options_never_install_or_download(self):
        with patch('subprocess.run') as process, patch('urllib.request.urlopen') as network:
            importlib.reload(optional_occlusion)
            MemoryOptions(occlusion=False).validate()
        process.assert_not_called()
        network.assert_not_called()

    def test_failed_install_preserves_environment_and_cleans_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('subprocess.run', side_effect=subprocess.CalledProcessError(1,'pip')) as process:
                with self.assertRaises(subprocess.CalledProcessError):
                    optional_occlusion.ensure_installed(directory)
            command=process.call_args.args[0]
            self.assertIn('--no-deps',command)
            self.assertIn('--target',command)
            self.assertNotIn('torch',command)
            self.assertEqual(process.call_args.kwargs['env']['SAM2_BUILD_CUDA'],'0')
            self.assertFalse((Path(directory)/'install.lock').exists())
            self.assertFalse((Path(directory)/'ready.json').exists())

    def test_concurrent_installer_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'install.lock').touch()
            with self.assertRaisesRegex(RuntimeError,'already running'):
                optional_occlusion.ensure_installed(directory)

    def test_tracker_exception_closes_process_and_parent_owned_temporary_files(self):
        from mosaichunk_h3.occlusion_tracking import OcclusionTracker
        from unittest.mock import Mock
        tracker=OcclusionTracker('unused')
        tracker.temp=tempfile.TemporaryDirectory()
        path=Path(tracker.temp.name)
        (path/'frame.jpg').write_bytes(b'private preview')
        tracker.process=Mock()
        tracker.process.poll.return_value=None
        tracker.__exit__(RuntimeError,RuntimeError('cancel'),None)
        tracker.process.terminate.assert_called_once()
        self.assertFalse(path.exists())


if __name__=='__main__':unittest.main()
