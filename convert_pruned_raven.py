"""Build compact RAVEN conditioning; never download the full H3 checkpoint.

The explicit command reads only pinned BF16 conditioning tensors with HTTP
Range requests. Each full-width projection lives in RAM for one iteration.
Only converted small projections are saved. Existing checkpoints are read-only.
"""

import argparse
import hashlib
import json
import math
import struct
import time
from pathlib import Path

import requests
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from mosaichunk_h3.pruned import fit_existing_time_curve, fold_projection, tensor_sha256

REVISION = "e5eb578a89295337b8ff433a035929ce0279e0b6"
URL = f"https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/{REVISION}/diffusion_models/minimax_h3_fl2va_bf16.safetensors"
ADAPTER_SHA256 = "99de2e6b1ff69c49c3ca4b1126e5679409037fd2ccd0442b8323dc310d328f30"
PREFIX = "base_model.model.dit."


class ConditioningReader:
    def __init__(self):
        self.session = requests.Session()
        self.bytes_read = 0
        n = struct.unpack("<Q", self.read_range(0, 8))[0]
        if n > 16 * 1024 * 1024:
            raise ValueError("Unexpected safetensors header length")
        raw = self.read_range(8, n)
        self.header_hash = hashlib.sha256(raw).hexdigest()
        if self.header_hash != "6e134a0085210edc037d8faf593a43d7ab9ad2019b58de0d23aa4652c014a6d5":
            raise ValueError("Pinned source header checksum mismatch")
        self.header = json.loads(raw)
        self.offset = n + 8

    def read_range(self, start, size):
        stop = start + size - 1
        for attempt in range(3):
            try:
                with self.session.get(URL + f"?conditioning_range={start}-{stop}",
                        headers={"Range": f"bytes={start}-{stop}", "Accept-Encoding": "identity"},
                        stream=True, timeout=(30, 180)) as response:
                    response.raise_for_status()
                    content_range = response.headers.get("Content-Range", "")
                    if response.status_code != 206 or not content_range.startswith(f"bytes {start}-{stop}/"):
                        raise ValueError("Server did not honor the exact byte range; refusing a full-file transfer")
                    data = bytearray()
                    for chunk in response.iter_content(8 * 1024 * 1024):
                        data.extend(chunk)
                        if len(data) > size:
                            raise ValueError("Server exceeded the requested range")
                    if len(data) != size:
                        raise requests.ConnectionError("Truncated tensor range")
                    self.bytes_read += len(data)
                    return data
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(2)

    def tensor(self, key, device):
        if not (key.startswith("time_embedder.") or ".adaln_proj.linear." in key):
            raise ValueError(f"Refusing to fetch non-conditioning tensor {key}")
        info = self.header[key]
        start, end = info["data_offsets"]
        dtype = {"BF16": torch.bfloat16, "F32": torch.float32}[info["dtype"]]
        data = self.read_range(self.offset + start, end - start)
        return torch.frombuffer(data, dtype=dtype).reshape(info["shape"]).to(device=device, copy=True)


def adapter_pair(reader, name, device):
    return tuple(reader.get_tensor(PREFIX + name + f".lora_{suffix}.weight").to(device)
                 for suffix in ("A", "B"))


def time_curve(t, weights, adapter):
    freqs = torch.exp(-math.log(10000.0) * torch.arange(128, device=t.device, dtype=torch.float32) / 128)
    angles = t[:, None] * freqs
    x = torch.cat((angles.cos(), angles.sin()), dim=-1)
    for name in ("proj_in", "proj_out"):
        a, b = adapter[name]
        x = F.linear(x, weights[name + ".weight"].float(), weights[name + ".bias"].float()) + F.linear(F.linear(x, a), b)
        x = F.silu(x)
    return x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path, help="Existing FL2VA pruned checkpoint (read-only)")
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inspect-only", action="store_true", help="Fetch header only and report the exact conditioning transfer size")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    digest = hashlib.sha256()
    with args.adapter.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != ADAPTER_SHA256:
        raise ValueError("Conversion requires the pinned published RAVEN adapter")
    with safe_open(args.base, framework="pt", device="cpu") as base:
        if "adaln_t_table" not in base.keys() or any("to_gate_compress" in key for key in base.keys()):
            raise ValueError("Use the existing FL2VA pruned model, not a FastVideo/VSA finetune")
        base_table = base.get_tensor("adaln_t_table")
        base_table_hash = tensor_sha256(base_table)
        if base_table_hash != "ac8727cdec52137c73878d004de5bd2a0e19227e8311e29ab3b68f328310e34e":
            raise ValueError("This converter requires the FL2VA pruned time table, not Ref2VA")
    reader = ConditioningReader()
    prefixes = [f"blocks.{i}" for i in range(50)] + ["final_layer"]
    required = [f"time_embedder.{n}.{s}" for n in ("proj_in", "proj_out") for s in ("weight", "bias")]
    required += [f"{p}.adaln_proj.linear.{s}" for p in prefixes for s in ("weight", "bias")]
    total = sum(reader.header[k]["data_offsets"][1] - reader.header[k]["data_offsets"][0] for k in required)
    print(json.dumps({"conditioning_transfer_bytes": total, "full_checkpoint_saved": False,
                      "source_revision": REVISION, "header_sha256": reader.header_hash}), flush=True)
    if args.inspect_only:
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    scratch = args.output.parent / (args.output.stem + ".conversion")
    scratch.mkdir(exist_ok=True)
    recipe = {"source_revision": REVISION, "source_header_sha256": reader.header_hash,
              "adapter_sha256": ADAPTER_SHA256, "base_table_sha256": base_table_hash,
              "rank": base_table.shape[1], "coordinate_basis": "original-base-table",
              "format": "mosaichunk-raven-conditioning-v1"}
    recipe_file = scratch / "recipe.json"
    if recipe_file.exists() and json.loads(recipe_file.read_text()) != recipe:
        raise ValueError("Existing partial conversion has a different recipe")
    recipe_file.write_text(json.dumps(recipe, indent=2))
    report = {"recipe": recipe, "layers": []}
    device = torch.device(args.device)
    with torch.inference_mode(), safe_open(args.adapter, framework="pt", device="cpu") as adapter:
        weights = {f"{n}.{s}": reader.tensor(f"time_embedder.{n}.{s}", device)
                   for n in ("proj_in", "proj_out") for s in ("weight", "bias")}
        time_adapter = {n: adapter_pair(adapter, "time_embedder." + n, device)
                        for n in ("proj_in", "proj_out")}
        curve = time_curve(torch.linspace(0, 1, 1025, device=device), weights, time_adapter)
        coords, basis, mean, residual = fit_existing_time_curve(curve, base_table)
        report["time_curve_relative_residual"] = residual
        if residual > 0.0001:
            raise ValueError(f"Time curve cannot be compressed safely: {residual}")
        # Off-grid probes include the actual four-step sigma grid's domain,
        # and both endpoints. The comparison uses the original BF16 lane.
        probes = torch.cat((torch.tensor([0., 1.], device=device),
                            torch.linspace(0.00037, 0.99963, 63, device=device)))
        probe_curve = time_curve(probes, weights, time_adapter)
        pos = probes * 1024
        i0 = pos.floor().long().clamp(max=1023)
        probe_coords = torch.lerp(coords[i0], coords[i0 + 1], (pos - i0).unsqueeze(1))
        tensors = {"adaln_t_table": coords.cpu()}
        for index, prefix in enumerate(prefixes):
            cached = scratch / f"layer-{index:02d}.safetensors"
            record_file = scratch / f"layer-{index:02d}.json"
            if cached.exists() and record_file.exists():
                tensors.update(load_file(str(cached)))
                report["layers"].append(json.loads(record_file.read_text()))
                print(f"Reused converted layer {index + 1}/51", flush=True)
                continue
            name = prefix + ".adaln_proj.linear"
            w = reader.tensor(name + ".weight", device)
            bias = reader.tensor(name + ".bias", device)
            a, b = adapter_pair(adapter, name, device)
            p, offset = fold_projection(w, bias, a, b, basis, mean)
            dense = F.linear(probe_curve.bfloat16(), w.bfloat16(), bias.bfloat16()).float()
            dense = (dense + F.linear(F.linear(probe_curve.bfloat16().float(), a), b)).bfloat16().float()
            folded = F.linear(probe_coords, p, offset)
            rel = float(torch.linalg.vector_norm(folded - dense) / torch.linalg.vector_norm(dense))
            if not math.isfinite(rel) or rel > 0.01:
                raise ValueError(f"Converted {name} differs from original BF16 RAVEN by {rel}; output refused")
            entry = {"layer": prefix, "offgrid_relative_l2_vs_bf16": rel}
            small = {name + ".weight": p.cpu(), name + ".bias": offset.cpu()}
            save_file(small, str(cached))
            record_file.write_text(json.dumps(entry))
            tensors.update(small)
            report["layers"].append(entry)
            print(f"Converted layer {index + 1}/51; relative L2 {rel:.6g}; transferred {reader.bytes_read / 1e9:.3f} GB", flush=True)
            del w, bias, a, b, p, offset, dense, folded
        report["transferred_bytes_this_run"] = reader.bytes_read
        report["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated() if device.type == "cuda" else None
        temp = args.output.with_suffix(".tmp.safetensors")
        save_file(tensors, str(temp), metadata={key: str(value) for key, value in recipe.items()})
        temp.replace(args.output)
        args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
        print(f"Saved compact conditioning: {args.output.stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
