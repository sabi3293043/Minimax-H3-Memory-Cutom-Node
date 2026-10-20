"""Per-evaluation allocation-compiler boundary for dynamic residual caches.

H3 starts allocation recording before dispatching DIFFUSION_MODEL wrappers.
End that recording before conditional block reuse. Prefetch/offload still runs;
no process-wide settings or ComfyUI functions are replaced. H3's final end call
is idempotent when the recording has already ended.
"""


class CacheCompilerBoundary:
    def __init__(self, session):
        self.session = session

    def __call__(self, executor, *args, **kwargs):
        import comfy.model_prefetch as prefetch
        streams = args[0] if args else kwargs.get('x')
        device = streams[0].device
        enabled = getattr(prefetch, 'malloc_graph_enabled', None)
        if enabled is not None and enabled(device):
            prefetch.malloc_graph_end()
            self.session.compiler_bypass_calls += 1
        return executor(*args, **kwargs)
