"""Structural patcher composition tests; not actual H3 / Spectrum quality tests."""
import sys
from pathlib import Path
from types import ModuleType
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mosaichunk_h3.native_visual_observer import MosaiChunkH3NativeVisualObserver, KEY


class FakePatcher:
    def __init__(self):
        self.wrappers = {
            "outer_sample": {"spectrum_h3": ["spectrum_wrapper"]},
            "predict_noise": {"spectrum_h3": ["predict_wrapper"]},
            "diffusion_model": {"spectrum_h3": ["diffusion_wrapper"]},
        }
        self.model_options = {
            "transformer_options": {"optimized_attention_override": object()}
        }

    def clone(self):
        other = FakePatcher()
        other.wrappers = {
            kind: {key: list(items) for key, items in group.items()}
            for kind, group in self.wrappers.items()
        }
        other.model_options = {
            "transformer_options": dict(self.model_options["transformer_options"])
        }
        return other

    def get_wrappers(self, kind, key):
        return self.wrappers.get(kind, {}).get(key, [])

    def add_wrapper_with_key(self, kind, key, wrapper):
        self.wrappers.setdefault(kind, {}).setdefault(key, []).append(wrapper)


class NativeCompositionTests(unittest.TestCase):
    def test_preserves_spectrum_wrappers_and_kj_sage_override(self):
        comfy = ModuleType("comfy")
        ext = ModuleType("comfy.patcher_extension")
        ext.WrappersMP = type("WrappersMP", (), {"OUTER_SAMPLE": "outer_sample"})
        comfy.patcher_extension = ext
        with patch.dict(sys.modules, {"comfy": comfy, "comfy.patcher_extension": ext}):
            model = FakePatcher()
            override = model.model_options["transformer_options"]["optimized_attention_override"]
            output, comparison = MosaiChunkH3NativeVisualObserver().apply(model, "auto")
            self.assertEqual(MosaiChunkH3NativeVisualObserver.RETURN_NAMES,('model','comparison'))
            self.assertIs(output.get_wrappers('outer_sample',KEY)[0].comparison_capture,comparison)
            self.assertIs(
                output.model_options["transformer_options"]["optimized_attention_override"],
                override,
            )
            self.assertEqual(output.wrappers["outer_sample"]["spectrum_h3"], ["spectrum_wrapper"])
            self.assertEqual(output.wrappers["predict_noise"]["spectrum_h3"], ["predict_wrapper"])
            self.assertEqual(output.wrappers["diffusion_model"]["spectrum_h3"], ["diffusion_wrapper"])
            self.assertIn(KEY, output.wrappers["outer_sample"])
            self.assertNotIn(KEY, model.wrappers["outer_sample"])
            with self.assertRaises(ValueError):
                MosaiChunkH3NativeVisualObserver().apply(output, "auto")

    def test_rejects_chunk_mode_controller(self):
        comfy = ModuleType("comfy")
        ext = ModuleType("comfy.patcher_extension")
        ext.WrappersMP = type("WrappersMP", (), {"OUTER_SAMPLE": "outer_sample"})
        comfy.patcher_extension = ext
        with patch.dict(sys.modules, {"comfy": comfy, "comfy.patcher_extension": ext}):
            model = FakePatcher()
            model.wrappers["outer_sample"]["mosaichunk_h3_controller"] = ["chunk_controller"]
            with self.assertRaisesRegex(ValueError, "RAVEN/MosaiChunk"):
                MosaiChunkH3NativeVisualObserver().apply(model, "auto")


if __name__ == "__main__":
    unittest.main()
