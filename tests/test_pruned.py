import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mosaichunk_h3.pruned import fit_time_curve, fit_existing_time_curve, fold_projection
from convert_pruned_raven import ConditioningReader, time_curve


class PrunedConversionTests(unittest.TestCase):
    def test_existing_lora_coordinate_convention_is_preserved(self):
        torch.manual_seed(23)
        original = torch.randn(101, 8)
        curve = original @ torch.randn(8, 32) + torch.randn(32)
        table, basis, mean, residual = fit_existing_time_curve(curve, original)
        self.assertTrue(torch.equal(table, original))
        self.assertLess(residual, 1e-6)
        torch.testing.assert_close(table @ basis + mean, curve, atol=3e-6, rtol=1e-4)

    def test_affine_fold_preserves_all_adapter_terms(self):
        torch.manual_seed(19)
        coords = torch.randn(101, 4)
        basis = torch.randn(4, 32)
        mean = torch.randn(32)
        x = coords @ basis + mean
        fitted, fitted_basis, fitted_mean, residual = fit_time_curve(x, 4)
        w, bias = torch.randn(73, 32), torch.randn(73)
        a, b = torch.randn(3, 32), torch.randn(73, 3)
        p, offset = fold_projection(w, bias, a, b, fitted_basis, fitted_mean, scale=0.7)
        expected = F.linear(x, w, bias) + 0.7 * F.linear(F.linear(x, a), b)
        actual = F.linear(fitted, p, offset)
        self.assertLess(residual, 1e-6)
        torch.testing.assert_close(actual, expected, atol=6e-5, rtol=1e-4)
        self.assertGreater(float((F.linear(fitted, p) - expected).norm()), 1)

    def test_time_adapter_changes_both_layers(self):
        torch.manual_seed(21)
        weights = {"proj_in.weight": torch.randn(12, 256), "proj_in.bias": torch.randn(12),
                   "proj_out.weight": torch.randn(7, 12), "proj_out.bias": torch.randn(7)}
        adapter = {"proj_in": (torch.randn(2, 256), torch.randn(12, 2)),
                   "proj_out": (torch.randn(2, 12), torch.randn(7, 2))}
        t = torch.tensor([0., 0.37, 1.])
        freq = torch.exp(-torch.log(torch.tensor(10000.)) * torch.arange(128) / 128)
        x = torch.cat(((t[:, None] * freq).cos(), (t[:, None] * freq).sin()), dim=-1)
        for name in ("proj_in", "proj_out"):
            a, b = adapter[name]
            x = F.silu(F.linear(x, weights[name + ".weight"] + b @ a, weights[name + ".bias"]))
        torch.testing.assert_close(time_curve(t, weights, adapter), x, atol=0.001, rtol=1e-4)

    def test_range_reader_refuses_full_response_before_reading(self):
        class Response:
            status_code = 200
            headers = {}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def raise_for_status(self): pass
            def iter_content(self, *args): raise AssertionError("Full model body must never be read")
        reader = ConditioningReader.__new__(ConditioningReader)
        reader.bytes_read = 0
        with patch("requests.Session.get", return_value=Response()):
            import requests
            reader.session = requests.Session()
            with self.assertRaisesRegex(ValueError, "refusing a full-file"):
                reader.read_range(0, 8)
        self.assertEqual(reader.bytes_read, 0)


if __name__ == "__main__":
    unittest.main()
