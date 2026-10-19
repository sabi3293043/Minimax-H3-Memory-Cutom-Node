"""Bounded, execution-local CPU residual storage with synchronous transfers.

Preserves tensor dtype. Transfers are deliberately ordered/blocking: CPU
buffers cannot be recycled while CUDA still reads them. No global streams.
"""
import torch


class HostAllocationError(MemoryError):
    pass


class HostResidual:
    @staticmethod
    def storage_bytes(shape, element_size, transfer_bytes):
        rows, width = shape
        chunk_rows = max(1, min(rows, transfer_bytes // (width * element_size)))
        return (rows + chunk_rows) * width * element_size

    def __init__(self, source, transfer_bytes):
        self.chunk_rows = max(1, min(source.shape[0], transfer_bytes //
                                    (source.shape[1] * source.element_size())))
        try:
            # Pin once at capture; never pin anew for every cache hit.
            self.data = torch.empty(source.shape, dtype=source.dtype, device='cpu',
                                    pin_memory=source.device.type == 'cuda')
            self.scratch = torch.empty((self.chunk_rows, source.shape[1]),
                                      dtype=source.dtype, device='cpu',
                                      pin_memory=source.device.type == 'cuda')
        except RuntimeError as exc:
            raise HostAllocationError(str(exc)) from exc
        self.nbytes = (self.data.numel() + self.scratch.numel()) * source.element_size()
        self.transferred = 0
        for first in range(0, len(source), self.chunk_rows):
            end = min(first + self.chunk_rows, len(source))
            self.data[first:end].copy_(source[first:end].detach(), non_blocking=False)
            if source.device.type == 'cuda':
                self.transferred += source[first:end].numel() * source.element_size()

    def finish(self, after):
        for first in range(0, len(after), self.chunk_rows):
            end = min(first + self.chunk_rows, len(after))
            scratch = self.scratch[:end - first]
            scratch.copy_(after[first:end].detach(), non_blocking=False)
            self.data[first:end].neg_().add_(scratch)
            if after.device.type == 'cuda':
                self.transferred += scratch.numel() * scratch.element_size()

    def add(self, current):
        # The result is the normal full block output; only cache staging is tiled.
        result = current.clone()
        for first in range(0, len(current), self.chunk_rows):
            end = min(first + self.chunk_rows, len(current))
            chunk = self.data[first:end].to(device=current.device, non_blocking=False)
            result[first:end].add_(chunk)
            if current.device.type == 'cuda':
                self.transferred += chunk.numel() * chunk.element_size()
            del chunk
        return result
