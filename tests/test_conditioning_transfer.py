import tempfile
from pathlib import Path
import unittest

import torch
from safetensors.torch import save_file

from mosaichunk_h3.conditioning_transfer import transfer_conditioning


class Target(torch.nn.Module):
    def __init__(self, table):
        super().__init__()
        self.register_buffer('adaln_t_table',table.clone())
        self.blocks=torch.nn.ModuleList([torch.nn.Module()])
        self.final_layer=torch.nn.Module()
        for parent,rows in ((self.blocks[0],24),(self.final_layer,16)):
            parent.adaln_proj=torch.nn.Module()
            parent.adaln_proj.linear=torch.nn.Linear(8,rows)


class ConditioningTransferTests(unittest.TestCase):
    def test_variant_base_is_preserved_under_affine_time_basis_change(self):
        rng=torch.Generator().manual_seed(13)
        source=torch.randn(1025,8,generator=rng)
        q=torch.linalg.qr(torch.randn(8,8,generator=rng)).Q
        table=source@q+torch.randn(8,generator=rng)
        target=Target(table)
        state={'adaln_t_table':source}
        adapted={'adaln_t_table':source}
        for name in ('blocks.0.adaln_proj.linear','final_layer.adaln_proj.linear'):
            layer=target.get_submodule(name)
            state[name+'.weight']=torch.randn(layer.weight.shape,generator=rng)
            state[name+'.bias']=torch.randn(layer.bias.shape,generator=rng)
            adapted[name+'.weight']=state[name+'.weight']+torch.randn(layer.weight.shape,generator=rng)*.01
            adapted[name+'.bias']=state[name+'.bias']+torch.randn(layer.bias.shape,generator=rng)*.01
        before={k:v.clone() for k,v in target.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'baseline.safetensors'
            save_file(state,str(path))
            output,error=transfer_conditioning(adapted,source,str(path),target)
        self.assertLess(error,1e-5)
        torch.testing.assert_close(output['adaln_t_table'],table,rtol=0,atol=0)
        # Probe between grid points, not only at the fitted table entries.
        s=.37*source[123:188]+.63*source[124:189]
        t=.37*table[123:188]+.63*table[124:189]
        for name in ('blocks.0.adaln_proj.linear','final_layer.adaln_proj.linear'):
            delta=s@(adapted[name+'.weight']-state[name+'.weight']).T+adapted[name+'.bias']-state[name+'.bias']
            expected=t@before[name+'.weight'].T+before[name+'.bias']+delta
            actual=t@output[name+'.weight'].T+output[name+'.bias']
            torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        for k,v in target.state_dict().items():
            torch.testing.assert_close(v,before[k],rtol=0,atol=0)

    def test_unrelated_time_basis_is_rejected_instead_of_silently_approximated(self):
        rng=torch.Generator().manual_seed(14)
        source=torch.randn(1025,8,generator=rng)
        target=Target(torch.randn(1025,8,generator=rng))
        state={'adaln_t_table':source}
        adapted={'adaln_t_table':source}
        for name in ('blocks.0.adaln_proj.linear','final_layer.adaln_proj.linear'):
            layer=target.get_submodule(name)
            state[name+'.weight']=torch.zeros_like(layer.weight)
            state[name+'.bias']=torch.zeros_like(layer.bias)
            adapted[name+'.weight']=torch.ones_like(layer.weight)
            adapted[name+'.bias']=torch.zeros_like(layer.bias)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'baseline.safetensors'
            save_file(state,str(path))
            with self.assertRaisesRegex(ValueError,'cannot be represented accurately'):
                transfer_conditioning(adapted,source,str(path),target)


if __name__=='__main__':
    unittest.main()
