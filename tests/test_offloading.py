import sys
import unittest
from types import SimpleNamespace
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mosaichunk_h3.runtime_linear import attach_residual


class OffloadingTests(unittest.TestCase):
    def test_prepared_dynamic_lane_propagates_native_prefetch_without_mutating_model_options(self):
        from mosaichunk_h3.contracts import resolve_transformer_options
        model_options = {"mosaichunk_dynamic_ready": True, "transformer_options": {}}
        patcher = SimpleNamespace(model_options=model_options, wrappers={}, is_dynamic=lambda: True)
        options = resolve_transformer_options(patcher)
        self.assertTrue(options["prefetch_dynamic_vbars"])
        self.assertEqual(model_options["transformer_options"], {})

    def test_adapter_conversion_occurs_once_per_forward_not_per_row_chunk(self):
        torch.manual_seed(10)
        linear = torch.nn.Linear(16, 8, bias=False)
        a = torch.randn(2, 16, dtype=torch.bfloat16)
        b = torch.randn(8, 2, dtype=torch.bfloat16)
        attach_residual(linear, a, b, path="test", alpha=2, rank=2, row_chunk=3)
        x = torch.randn(23, 16)

        class Copies(TorchDispatchMode):
            def __init__(self):
                super().__init__()
                self.count = 0
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if func == torch.ops.aten._to_copy.default and tuple(args[0].shape) in ((2, 16), (8, 2)):
                    self.count += 1
                return func(*args, **(kwargs or {}))

        with Copies() as copies, torch.no_grad():
            actual = linear(x)
        expected = torch.nn.functional.linear(x, linear.weight) + (x @ a.float().T) @ b.float().T
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
        self.assertEqual(copies.count, 2)
        self.assertEqual(linear.raven_lora_A_0.dtype, torch.bfloat16)
        self.assertEqual(linear.raven_lora_B_0.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
