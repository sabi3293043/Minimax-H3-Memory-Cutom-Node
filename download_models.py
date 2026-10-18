"""Explicit, resumable downloads of the pinned H3/MosaiChunk model artifacts."""

import argparse
import hashlib
import os
from pathlib import Path
import shutil

ARTIFACTS = [
    ("router", "mosaichunk/MosaiChunk", "03eb1531e2363cbee4fe27d773c4242a07eb1b32",
     "t2v/config.json", "mosaichunk/t2v/config.json", 653,
     "a89f21524f2efa25e83b7b7f45fe6645b8754f02405c5520bba5d6cb84d83a2f"),
    ("router", "mosaichunk/MosaiChunk", "03eb1531e2363cbee4fe27d773c4242a07eb1b32",
     "t2v/model.safetensors", "mosaichunk/t2v/model.safetensors", 54596844,
     "ddbb2449f0046985e9f3bd17d8fa39d9dc1f25cf31115b6713477e63895b44a3"),
    ("adapter", "mvp-lab/MiniMax-H3-RAVEN-Streaming-LoRA", "90f30b11639e73ce0e3f2ab6ed70d1a133a66caf",
     "minimax_h3_raven_streaming_lora_4nfe_preview.safetensors",
     "loras/minimax_h3_raven_streaming_lora_4nfe_preview.safetensors", 5057484112,
     "99de2e6b1ff69c49c3ca4b1126e5679409037fd2ccd0442b8323dc310d328f30"),
    ("base", "Comfy-Org/MiniMax-H3", "e5eb578a89295337b8ff433a035929ce0279e0b6",
     "diffusion_models/minimax_h3_fl2va_bf16.safetensors",
     "diffusion_models/minimax_h3_fl2va_bf16.safetensors", 66280487368,
     "907d4add438438ec1544f5240c3b38532ed934fe6be75677a6bbda2a6fdd6182"),
]


def verify(path, expected_size, expected_hash):
    if path.stat().st_size != expected_size:
        raise ValueError(f"Unexpected file size: {path}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != expected_hash:
        raise ValueError(f"SHA256 mismatch: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--components", choices=("router", "adapter", "base"), nargs="+",
                        default=["router", "adapter", "base"])
    parser.add_argument("--plan", action="store_true", help="Show files/sizes without downloading.")
    parser.add_argument("--verify", action="store_true", help="Also hash existing files (reads 66 GB for the base).")
    args = parser.parse_args()
    root = args.comfy_root.resolve()
    if not (root / "folder_paths.py").is_file() or not (root / "models").is_dir():
        parser.error("--comfy-root must point to the ComfyUI folder containing folder_paths.py and models/.")
    chosen = [entry for entry in ARTIFACTS if entry[0] in args.components]
    missing = []
    for component, repo, revision, name, dest, size, digest in chosen:
        target = root / "models" / dest
        exists = target.is_file() and target.stat().st_size == size
        print(f"{'EXISTS' if exists else 'DOWNLOAD'} {size / 1e9:.3f} GB  models/{dest}", flush=True)
        if exists:
            if args.verify:
                verify(target, size, digest)
        else:
            missing.append((component, repo, revision, name, target, size, digest))
    if args.plan or not missing:
        return
    required = sum(entry[5] for entry in missing)
    if shutil.disk_usage(root).free < required + 2 * 1024 ** 3:
        raise RuntimeError(f"Need at least {required / 1e9:.1f} GB free plus 2 GiB working space on the model drive.")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ.setdefault("HF_XET_CACHE", str(root / "models/.mosaichunk-download/xet"))
    from huggingface_hub import hf_hub_download
    for component, repo, revision, name, target, size, digest in missing:
        stage = root / "models/.mosaichunk-download" / component
        print(f"Downloading {repo}/{name}", flush=True)
        downloaded = Path(hf_hub_download(repo, name, revision=revision, local_dir=stage))
        verify(downloaded, size, digest)
        if target.exists():
            raise FileExistsError(f"Preserving the existing file at {target}; move it aside before retrying.")
        target.parent.mkdir(parents=True, exist_ok=True)
        downloaded.replace(target)
        print(f"Verified {target.name}", flush=True)


if __name__ == "__main__":
    main()
