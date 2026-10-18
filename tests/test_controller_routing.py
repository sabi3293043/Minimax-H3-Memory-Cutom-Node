import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from mosaichunk_h3.controller import FL2VA_TABLE, select_mode


class ControllerRoutingTests(unittest.TestCase):
    def setUp(self):
        self.root = SimpleNamespace(adaln_t_table=torch.zeros(1025, 8))

    def test_supported_text_only_checkpoint_activates_memory(self):
        with patch('mosaichunk_h3.controller.tensor_sha256', return_value=FL2VA_TABLE):
            self.assertEqual(select_mode({}, {'positive': [{}]}, self.root), 'text_to_video')

    def test_image_and_reference_rows_override_explicit_text_mode(self):
        for key, expected in [('minimax_keyframes', 'first_last_frame'), ('minimax_refs', 'reference')]:
            with self.subTest(key=key):
                # A tensor extra must not trigger ambiguous tensor truth values.
                conds = {'positive': [{key: torch.ones(2, 3)}]}
                self.assertEqual(select_mode({'generation_mode': 'text_to_video'}, conds, self.root), expected)

    def test_explicit_inactive_modes_never_read_model_table(self):
        for mode in ('off', 'first_last_frame', 'reference'):
            with self.subTest(mode=mode), patch('mosaichunk_h3.controller.tensor_sha256') as digest:
                self.assertEqual(select_mode({'generation_mode': mode}, {}, self.root), mode)
                digest.assert_not_called()

    def test_nonmatching_checkpoint_remains_native(self):
        self.assertEqual(select_mode({}, {}, self.root), 'unsupported_checkpoint')
        self.assertEqual(select_mode({}, {}, object()), 'unsupported_checkpoint')

    def test_invalid_mode_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, 'Unknown H3 controller mode'):
            select_mode({'generation_mode': 'typo'}, {}, self.root)


if __name__ == '__main__':
    unittest.main()
