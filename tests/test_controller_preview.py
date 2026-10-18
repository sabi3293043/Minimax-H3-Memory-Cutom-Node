"""Exercise the installed KJ callback wrapper through native CFGGuider on CPU."""
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import torch

if os.environ.get('COMFYUI_PATH'):
    sys.path.insert(0, os.environ['COMFYUI_PATH'])
try:
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    import comfy.latent_formats
    import comfy.model_patcher
    import comfy.samplers
    import comfy.utils
    from comfy.patcher_extension import WrappersMP
except ImportError:
    raise unittest.SkipTest('Native controller tests need ComfyUI')

from mosaichunk_h3.controller import attach_controller, MemoryController, strip_controller
from mosaichunk_h3.contracts import ContractError, resolve_transformer_options


class PreviewCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = Path(comfy.cli_args.__file__).resolve().parents[1]
        directory = app / 'custom_nodes/ComfyUI-KJNodes/nodes'
        source = directory / 'preview_override_node.py'
        if not source.exists():
            raise unittest.SkipTest('KJ preview node is not installed')
        # Load only the real preview module and its tiny decoder helper, without
        # importing the entire unrelated KJ node registry or optional LTX backend.
        package = ModuleType('_mosaichunk_kj_preview_test')
        package.__path__ = [str(directory)]
        name = package.__name__ + '.preview_override_node'
        spec = importlib.util.spec_from_file_location(name, source)
        cls.preview_module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {package.__name__: package,
                                     package.__name__ + '.ltxv_nodes': None,
                                     name: cls.preview_module}):
            spec.loader.exec_module(cls.preview_module)

    def setUp(self):
        base = torch.nn.Module()
        base.diffusion_model = torch.nn.Module()
        base.diffusion_model.adaln_t_table = torch.zeros(1025, 8)
        base.latent_format = comfy.latent_formats.MiniMaxH3AV()
        base.get_dtype = lambda: torch.float32
        self.source = comfy.model_patcher.ModelPatcher(base, torch.device('cpu'), torch.device('cpu'))
        self.controls = dict(generation_mode='auto', memory_mode='mosaichunk',
                             offloading='inherit', kv_cache_storage='cpu', far_memory_chunks=1)
        self.preview = self.preview_module._PreviewOverrideWrapper(64, None, 80, True)

    def controller(self, model):
        return attach_controller(model, 'conditioning', 'adapter', self.controls, Mock())

    def preview_model(self, model):
        model = model.clone()
        model.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, 'kj_preview_override', self.preview)
        return model

    def run_preview(self, preview_first, fail=False):
        model = (self.controller(self.preview_model(self.source)) if preview_first
                 else self.preview_model(self.controller(self.source)))
        guider = comfy.samplers.CFGGuider(model)
        guider.set_conds([], [])
        guider.set_cfg(1.)
        callback = Mock()
        solver = Mock()
        noise = torch.randn(1, 24, 1, 4, 4)
        original_decode = self.preview_module.latent_preview.LatentPreviewer.decode_latent_to_preview_image

        def memory(guider, mode, clean, noise, latent, sampler, sigmas,
                   mask, callback, disable_pbar, seed, latent_shapes=None):
            self.assertEqual(mode, 'text_to_video')
            self.assertFalse(clean.wrappers)
            resolve_transformer_options(clean)
            for step in range(6):
                callback(step, noise, noise, 6)
            if fail:
                raise RuntimeError('memory-test-cancel')
            return noise

        with patch.object(MemoryController, 'sample_memory', side_effect=memory) as run, \
             patch.object(self.preview_module, 'PromptServer', None):
            if fail:
                with self.assertRaisesRegex(RuntimeError, 'memory-test-cancel'):
                    guider.sample(noise, torch.zeros_like(noise), solver,
                                  torch.tensor([1., 0.]), callback=callback, seed=7)
            else:
                output = guider.sample(noise, torch.zeros_like(noise), solver,
                                       torch.tensor([1., 0.]), callback=callback, seed=7)
                torch.testing.assert_close(output, noise)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(callback.call_count, 6)
        self.assertEqual(len(self.preview.frames), 6)
        self.assertIs(original_decode, self.preview_module.latent_preview.LatentPreviewer.decode_latent_to_preview_image)
        self.assertIn('kj_preview_override', model.wrappers[WrappersMP.OUTER_SAMPLE])
        self.assertFalse(self.source.wrappers)
        solver.sample.assert_not_called()

    def test_preview_after_controller(self):
        self.run_preview(False)

    def test_preview_before_controller_runs_once(self):
        self.run_preview(True)

    def test_preview_cleanup_on_memory_failure(self):
        self.run_preview(False, fail=True)

    def test_unknown_wrapper_is_not_silently_discarded(self):
        model = self.preview_model(self.controller(self.source))
        unknown = Mock()
        model.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, 'unknown_feature', unknown)
        clean = strip_controller(model, handled_previews=True)
        with self.assertRaisesRegex(ContractError, 'unknown_feature'):
            resolve_transformer_options(clean)
        self.assertIn('kj_preview_override', model.wrappers[WrappersMP.OUTER_SAMPLE])
        unknown.assert_not_called()

    def test_preview_key_does_not_whitelist_unknown_function(self):
        model = self.controller(self.source)
        model.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, 'kj_preview_override', Mock())
        with self.assertRaisesRegex(ContractError, 'kj_preview_override'):
            resolve_transformer_options(strip_controller(model, handled_previews=True))
