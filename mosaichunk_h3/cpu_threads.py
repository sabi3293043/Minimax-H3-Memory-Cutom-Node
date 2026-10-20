"""Resolve isolated tracker threads without importing torch or changing the host."""
import os


def available_cpu_threads():
    process_count = getattr(os, 'process_cpu_count', None)
    if process_count is not None:
        count = process_count()
        if count:
            return max(1, count)
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    if os.name == 'nt':
        try:
            import ctypes
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.GetCurrentProcess.restype = ctypes.c_void_p
            kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
            process, system = ctypes.c_size_t(), ctypes.c_size_t()
            if kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(),
                    ctypes.byref(process), ctypes.byref(system)):
                return max(1, process.value.bit_count())
        except (AttributeError, OSError):
            pass
    return max(1, os.cpu_count() or 1)


def validate_cpu_threads(requested):
    if isinstance(requested, bool) or not isinstance(requested, int) or not 0 <= requested <= 256:
        raise ValueError('occlusion_cpu_threads must be an integer from 0 to 256; 0 uses all available threads')
    return requested


def resolve_cpu_threads(requested):
    validate_cpu_threads(requested)
    available = available_cpu_threads()
    return available if requested == 0 else min(requested, available)
