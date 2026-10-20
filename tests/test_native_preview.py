"""Standalone preview consumes CPU RGB; bounded encoding and explicit external ownership."""
import base64
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from mosaichunk_h3.native_preview import MemoryPreview,RepairComparison,has_external_preview,external_preview_uses_decoder,select_indices


class PreviewTests(unittest.TestCase):
    def test_external_preview_detects_both_native_wrapper_registries(self):
        wrappers={'outer_sample':{'kj_preview_override':[object()]}}
        self.assertFalse(has_external_preview(None))
        self.assertTrue(has_external_preview(SimpleNamespace(model_options={'transformer_options':{'wrappers':wrappers}})))
        self.assertTrue(has_external_preview(SimpleNamespace(model_patcher=SimpleNamespace(wrappers=wrappers))))

    def test_external_resolution_is_followed_only_for_the_same_decoder_file(self):
        decoder=SimpleNamespace(_minimax_tiny_vae_name='taeh3.safetensors')
        wrapper=SimpleNamespace(tiny_vae='alias.safetensors')
        guider=SimpleNamespace(model_patcher=SimpleNamespace(wrappers={
            'outer_sample':{'kj_preview_override':[wrapper]}}))
        with patch('folder_paths.get_full_path',return_value='same.safetensors'):
            self.assertTrue(external_preview_uses_decoder(guider,decoder))
        with patch('folder_paths.get_full_path',side_effect=['first.safetensors','second.safetensors']):
            self.assertFalse(external_preview_uses_decoder(guider,decoder))

    def test_preview_selects_bounded_frames_and_encodes_valid_jpegs(self):
        self.assertEqual(select_indices(100,3),(0,50,99))
        frames=torch.full((3,3,24,32),120,dtype=torch.uint8)
        for encoded in MemoryPreview.encode(frames,16):
            image=Image.open(io.BytesIO(base64.b64decode(encoded)))
            self.assertEqual(image.size,(16,12))

    def test_background_encoding_delivers_owned_cpu_frames_and_stops(self):
        frames=torch.full((5,3,24,32),120,dtype=torch.uint8)
        with patch('mosaichunk_h3.native_preview.send') as send:
            with MemoryPreview('5',enabled=True,frames=3) as preview:
                preview.submit(frames,step=4,total=20)
            self.assertFalse(preview.thread.is_alive())
            payload=send.call_args.kwargs
            self.assertEqual(len(payload['frames']),3)
            self.assertEqual(payload['step'],5)
            self.assertEqual(payload['indices'],(0,2,4))

    def test_repair_comparison_reuses_bounded_aligned_frames_and_keeps_rejected_attempt(self):
        before=torch.zeros(17,3,24,32,dtype=torch.uint8)
        after=torch.full_like(before,200)
        with patch('mosaichunk_h3.native_preview.send') as send,patch.object(MemoryPreview,'encode',wraps=MemoryPreview.encode) as encode:
            preview=RepairComparison('5',enabled=True,frames=3,max_resolution=16)
            preview.start(before)
            initial=send.call_args.kwargs
            preview.finish(after,outcome='rejected',caption='Original retained')
            final=send.call_args.kwargs
        self.assertEqual(encode.call_count,2)
        self.assertEqual(initial['indices'],(0,8,16))
        self.assertEqual(initial['comparison']['phase'],'inspecting')
        self.assertIsNone(initial['comparison']['after'])
        self.assertEqual(final['comparison']['before'],initial['comparison']['before'])
        self.assertEqual(final['comparison']['outcome'],'rejected')
        self.assertEqual(len(final['comparison']['before']),3)
        self.assertEqual(len(final['comparison']['after']),3)
        for key in ('before','after'):
            image=Image.open(io.BytesIO(base64.b64decode(final['comparison'][key][0])))
            self.assertEqual(image.size,(16,12))
            self.assertEqual(image.getpixel((4,4)),(0,0,0) if key=='before' else (200,200,200))

    def test_skipped_repair_never_encodes_an_after_view(self):
        with patch('mosaichunk_h3.native_preview.send') as send,patch.object(MemoryPreview,'encode',return_value=['jpeg']) as encode:
            preview=RepairComparison('5',enabled=True,frames=1)
            preview.start(torch.zeros(3,3,4,4))
            preview.finish(outcome='skipped',caption='Original retained')
        self.assertEqual(encode.call_count,1)
        self.assertIsNone(send.call_args.kwargs['comparison']['after'])
        self.assertEqual(send.call_args.kwargs['comparison']['outcome'],'skipped')

    def test_comparison_off_encodes_and_sends_nothing(self):
        with patch('mosaichunk_h3.native_preview.send') as send,patch.object(MemoryPreview,'encode') as encode:
            for node,enabled in (('5',False),(None,True)):
                preview=RepairComparison(node,enabled=enabled)
                preview.start(torch.zeros(3,3,4,4))
                preview.finish(outcome='skipped',caption='Retained')
        send.assert_not_called();encode.assert_not_called()

    def test_comparison_geometry_failure_does_not_misrepresent_the_saved_result(self):
        with patch('mosaichunk_h3.native_preview.send') as send:
            preview=RepairComparison('5',enabled=True)
            preview.start(torch.zeros(3,3,4,4))
            with self.assertLogs('mosaichunk_h3.native_preview',level='WARNING'):
                preview.finish(torch.zeros(2,3,4,4),outcome='accepted',caption='Repair accepted')
        payload=send.call_args.kwargs['comparison']
        self.assertEqual(payload['outcome'],'accepted')
        self.assertTrue(payload['unavailable'])
        self.assertIsNone(payload['after'])

    def test_disabled_preview_does_not_start_encoder(self):
        with patch.object(MemoryPreview,'encode') as encode:
            with MemoryPreview('5',enabled=False) as preview:
                preview.submit(torch.zeros(2,3,4,4),step=0,total=20)
            self.assertIsNone(preview.thread)
            encode.assert_not_called()


if __name__=='__main__':unittest.main()
