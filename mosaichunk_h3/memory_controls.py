"""UI-independent options and early compatibility diagnostics."""
from dataclasses import dataclass

from .region_memory import parse_regions


@dataclass(frozen=True)
class MemoryOptions:
    regions: str = '[]'
    protected_frames: str = ''
    inspect: bool = False
    export_report: bool = False
    occlusion: bool = False
    preset: str = 'Custom'

    def validate(self):
        from .native_guidance import frame_numbers
        parse_regions(self.regions)
        if len(frame_numbers(self.protected_frames)) > 8:
            raise ValueError('Protect at most eight frames')
        if self.preset not in ('Consistency', 'Balanced', 'Speed', 'Custom'):
            raise ValueError('Unknown memory preset')
        return self


def compatibility(guider, *, guided, accelerated, denoise_mask):
    """Checks with actionable messages before sampling begins."""
    if not hasattr(guider, 'model_options'):
        raise ValueError('MiniMax H3 Memory needs an existing native BasicGuider/CFGGuider sampler')
    if (guided or accelerated) and float(getattr(guider,'cfg',1.)) != 1.:
        raise ValueError('MiniMax H3 Memory guidance/residual caching requires CFG=1. Set CFG on your guider.')
    if (guided or accelerated) and denoise_mask is not None:
        raise ValueError('MiniMax H3 Memory guidance/residual caching does not support a denoise mask. Disconnect it or use observe_only with acceleration off.')
