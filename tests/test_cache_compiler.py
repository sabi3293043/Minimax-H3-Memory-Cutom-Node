"""Local compiler opt-out preserves native execution and option restoration."""
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import test_generation_acceleration as helpers
from mosaichunk_h3.cache_compiler import CacheCompilerBoundary
from mosaichunk_h3.generation_acceleration import install_acceleration


class BoundaryTests(unittest.TestCase):
    def test_ends_active_recording_before_network_and_preserves_arguments(self):
        comfy = ModuleType('comfy')
        prefetch = ModuleType('comfy.model_prefetch')
        comfy.model_prefetch = prefetch
        calls = []
        prefetch.malloc_graph_enabled = lambda d: d == 'cuda'
        prefetch.malloc_graph_end = lambda: calls.append('end')
        session = SimpleNamespace(compiler_bypass_calls=0)
        wrapper = CacheCompilerBoundary(session)
        def network(*args, **kw):
            calls.append('network')
            self.assertIs(args[0], streams)
            self.assertEqual(kw, {'seed': 11})
            return 'result'
        streams = [SimpleNamespace(device='cuda')]
        with patch.dict('sys.modules', {'comfy': comfy, 'comfy.model_prefetch': prefetch}):
            self.assertEqual(wrapper(network, streams, seed=11), 'result')
        self.assertEqual(calls, ['end', 'network'])
        self.assertEqual(session.compiler_bypass_calls, 1)

    def test_inactive_compiler_does_not_end_any_recording(self):
        comfy = ModuleType('comfy')
        prefetch = ModuleType('comfy.model_prefetch')
        comfy.model_prefetch = prefetch
        prefetch.malloc_graph_enabled = lambda d: False
        prefetch.malloc_graph_end = lambda: self.fail('inactive compiler')
        session = SimpleNamespace(compiler_bypass_calls=0)
        with patch.dict('sys.modules', {'comfy': comfy, 'comfy.model_prefetch': prefetch}):
            out = CacheCompilerBoundary(session)(lambda *a: 19, [SimpleNamespace(device='cpu')])
        self.assertEqual(out, 19)
        self.assertEqual(session.compiler_bypass_calls, 0)


class ScopeTests(unittest.TestCase):
    setUp = helpers.AccelerationTests.setUp

    def test_existing_diffusion_wrappers_are_retained_and_source_options_restored(self):
        original_wrapper = lambda executor,*a,**k: executor(*a,**k)
        self.options['transformer_options']['wrappers'] = {'diffusion_model': {'third_party': [original_wrapper]}}
        with install_acceleration(self.guider, self.config, self.guidance):
            groups = self.guider.model_options['transformer_options']['wrappers']['diffusion_model']
            self.assertEqual(list(groups), ['mosaichunk_cache_allocation_boundary','third_party'])
            self.assertIs(groups['third_party'][0], original_wrapper)
        self.assertIs(self.guider.model_options, self.options)
        self.assertEqual(list(self.options['transformer_options']['wrappers']['diffusion_model']), ['third_party'])
