"""Execution-local composition tests. Actual Comfy/H3 is checked separately."""
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from mosaichunk_h3.native_guidance import GuidanceConfig, GuidanceSession, install_guidance


def copy_options(value):
    if isinstance(value, dict):
        return {key: copy_options(item) for key, item in value.items()}
    if isinstance(value, list):
        return list(value)
    return value


class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.ext = ModuleType('comfy.patcher_extension')
        self.ext.copy_nested_dicts = copy_options
        self.model = SimpleNamespace(blocks=[SimpleNamespace(attn=SimpleNamespace(comfy_attention=SimpleNamespace(function=None)))])
        self.patcher = SimpleNamespace(get_model_object=lambda name: self.model)
        self.session = GuidanceSession(GuidanceConfig(mode='manual_frames', layers='0',
            target_frames='2'), [1, 24, 3, 4, 4])

    def test_original_options_sage_and_block_replacements_are_preserved(self):
        calls = []
        sage = lambda *args, **kwargs: 'sage'

        def previous(args, extra):
            calls.append('previous')
            self.assertIsNot(args['transformer_options']['optimized_attention_override'], sage)
            return extra['original_block'](args)

        original = {'transformer_options': {'optimized_attention_override': sage,
            'patches_replace': {'dit': {('double_block', 0): previous}}}}
        guider = SimpleNamespace(model_options=original, model_patcher=self.patcher)
        with patch.dict('sys.modules', {'comfy.patcher_extension': self.ext}):
            with install_guidance(guider, self.session):
                replacements = guider.model_options['transformer_options']['patches_replace']['dit']
                def block(args):
                    override = args['transformer_options']['optimized_attention_override']
                    with patch.object(self.session, 'attention', side_effect=lambda backend, *a, **kw: backend(*a, **kw)):
                        result = override(lambda: 'unused', None, None, None, 2)
                    return {'img': result}
                out = replacements[('double_block', 0)]({'transformer_options': original['transformer_options']}, {'original_block': block})
                self.assertEqual(out, {'img': 'sage'})
        self.assertEqual(calls, ['previous'])
        self.assertIs(guider.model_options, original)
        self.assertIs(original['transformer_options']['optimized_attention_override'], sage)
        self.assertIs(original['transformer_options']['patches_replace']['dit'][('double_block', 0)], previous)

    def test_scope_restores_options_and_releases_pairs_after_failure(self):
        original = {'transformer_options': {}}
        guider = SimpleNamespace(model_options=original, model_patcher=self.patcher)
        self.session.pairs = ((0, 2),)
        with patch.dict('sys.modules', {'comfy.patcher_extension': self.ext}):
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                with install_guidance(guider, self.session):
                    raise RuntimeError('interrupted')
        self.assertIs(guider.model_options, original)
        self.assertFalse(self.session.pairs)
        self.assertEqual(original, {'transformer_options': {}})

    def test_spectrum_uses_independent_actual_runtime(self):
        @dataclass(frozen=True)
        class Config:
            force_actual: bool = False
            offline_smoothing_replay: bool = True
            anchor_residual_feedback: bool = False
            selective_rollback_correction: bool = False
            model_aware_mode: str = 'off'
            model_aware_trust_shrinkage: bool = False
            model_aware_replay_generic_correction: bool = False
        @dataclass
        class Runtime:
            config: Config
        @dataclass
        class Binding:
            runtime: Runtime
        source = Binding(Runtime(Config()))
        original = {'transformer_options': {}, 'spectrum_h3_binding': source}
        guider = SimpleNamespace(model_options=original, model_patcher=self.patcher)
        with patch.dict('sys.modules', {'comfy.patcher_extension': self.ext}):
            with install_guidance(guider, self.session):
                active = guider.model_options['spectrum_h3_binding'].runtime
                self.assertTrue(active.config.force_actual)
                self.assertFalse(active.config.offline_smoothing_replay)
                self.assertIsNot(active, source.runtime)
        self.assertFalse(source.runtime.config.force_actual)
        self.assertTrue(source.runtime.config.offline_smoothing_replay)


if __name__ == '__main__':
    unittest.main()
