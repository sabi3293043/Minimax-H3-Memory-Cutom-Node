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
    cpu_threads: int = 0
    share_frames: bool = True
    preview: str = 'auto'
    preview_frames: int = 16
    preview_resolution: int = 512
    preview_fps: float = 8.
    tracking_selection: str = 'retrieved'
    tracking_frame_budget: int = 8

    def validate(self):
        from .native_guidance import frame_numbers
        from .cpu_threads import validate_cpu_threads
        validate_cpu_threads(self.cpu_threads)
        if self.tracking_selection not in ('retrieved', 'uniform'):
            raise ValueError('Unknown occlusion_frame_selection mode')
        if (isinstance(self.tracking_frame_budget, bool) or not isinstance(self.tracking_frame_budget, int)
                or not 4 <= self.tracking_frame_budget <= 16):
            raise ValueError('occlusion_frame_budget must be an integer from 4 to 16')
        import math
        if self.preview not in ('auto','off'):
            raise ValueError('Unknown memory_preview mode')
        if not isinstance(self.preview_frames,int) or not 1<=self.preview_frames<=64:
            raise ValueError('memory_preview_frames must be 1..64')
        if not isinstance(self.preview_resolution,int) or not 64<=self.preview_resolution<=1024:
            raise ValueError('memory_preview_resolution must be 64..1024')
        if not math.isfinite(self.preview_fps) or not 1<=self.preview_fps<=30:
            raise ValueError('memory_preview_fps must be 1..30')
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
