"""Local ComfyUI nodes for the MosaiChunk T2V router and RAVEN sampler."""

import json
from pathlib import Path

import folder_paths
import comfy.utils
import torch

from .memory import MemorySettings, MosaiChunkKVCache
from .nodes import RAVENModelLoader, RAVENStreamingSampler
from .controller import attach_controller, MODES
from .routing.selector import Selector

ROUTER_FOLDER = "mosaichunk"
folder_paths.folder_names_and_paths.setdefault(ROUTER_FOLDER, ([], {".safetensors"}))
folder_paths.add_model_folder_path(ROUTER_FOLDER, str(Path(folder_paths.models_dir) / ROUTER_FOLDER))


class MosaiChunkH3ModelLoader(RAVENModelLoader):
    CATEGORY = "MiniMax H3/MosaiChunk"
    DESCRIPTION = "Load the full non-pruned H3 model with the mandatory RAVEN streaming adapter."


class MosaiChunkH3Conditioning:
    CATEGORY = "MiniMax H3/MosaiChunk"
    DESCRIPTION = (
        "MODEL-to-MODEL controller for ordinary H3 samplers. Active memory uses the RAVEN "
        "chunk consistency recipe with sampler seed, initial noise and step count; "
        "the selected solver and sigma values are replaced. Use four steps/CFG 1. "
        "First/last-frame and reference modes retain their image/audio context in "
        "the permanent memory prefix. Image modes and REF2VA adapter transfer are "
        "experimental. Offloading inherits the incoming native policy."
    )
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "conditioning_name": (folder_paths.get_filename_list(ROUTER_FOLDER),
                                  {"default": "raven-fl2va-conditioning.safetensors"}),
            "raven_lora": (folder_paths.get_filename_list("loras"),
                           {"default": "minimax_h3_raven_streaming_lora_4nfe_preview.safetensors",
                            "tooltip": "Mandatory RAVEN adapter. Keep your other LoRAs on the incoming MODEL chain."}),
            "video_shift": ("FLOAT", {"default": 12.0, "min": 0.01, "max": 30.0, "step": 0.1}),
            "audio_shift": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 30.0, "step": 0.1}),
            "kv_cache_storage": (["cpu", "cpu_pinned"], {"default": "cpu"}),
            "far_memory_chunks": ("INT", {"default": 1, "min": 1, "max": 2}),
            "history_ram_gib": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 256.0, "step": 1.0}),
            "memory_mode": (["mosaichunk", "matched_window"], {"default": "mosaichunk"}),
            "offloading": (["inherit", "static"], {"default": "inherit",
                           "tooltip": "inherit keeps the incoming ComfyUI dynamic/static offloader; static is an explicit fallback."}),
        }, "optional": {
            "generation_mode": (list(MODES), {"default": "auto"}),
            "router_name": (router_names(), {"default": "t2v/model.safetensors"}),
            "conditioning_base_unet": (["auto"] + folder_paths.get_filename_list("diffusion_models"),
                {"default": "auto", "tooltip": "Original FL2VA pruned checkpoint used ONLY to transfer the RAVEN modulation update onto the input model. Auto uses your existing standard FL2VA INT8 file. No download."}),
        }, "hidden": {"unique_id": "UNIQUE_ID"}}

    def apply(self, model, conditioning_name, raven_lora, video_shift=12., audio_shift=3.,
              kv_cache_storage="cpu", far_memory_chunks=1, history_ram_gib=8., memory_mode="mosaichunk", offloading="inherit",
              generation_mode="auto", router_name="t2v/model.safetensors", conditioning_base_unet="auto", unique_id=None):
        conditioning_path = folder_paths.get_full_path(ROUTER_FOLDER, conditioning_name)
        adapter_path = folder_paths.get_full_path("loras", raven_lora)
        controls = dict(video_shift=float(video_shift), audio_shift=float(audio_shift),
                        kv_cache_storage=kv_cache_storage, far_memory_chunks=int(far_memory_chunks),
                        history_ram_gib=float(history_ram_gib), memory_mode=memory_mode, offloading=offloading,
                        generation_mode=generation_mode, conditioning_base_unet=conditioning_base_unet)
        # Resolve/load artifacts only if the runtime actually selects active memory.
        def cache_builder(prepared):
            return build_cache_factory(prepared, router_name, controls)
        return (attach_controller(model, conditioning_path, adapter_path, controls, cache_builder, unique_id),)


class MosaiChunkH3StreamingSampler(RAVENStreamingSampler):
    CATEGORY = "MiniMax H3/MosaiChunk"
    DESCRIPTION = (
        "Experimental MosaiChunk T2VA port: two recent chunks plus one/two chunks of retrieved visual memory. "
        "History stays in system RAM; GPU memory is assembled one layer at a time. "
        "Use a RAVEN-conditioned H3 MODEL, an empty AV latent and text-only conditioning. "
        "Start at 512x320 / 90 frames / 4 steps / CPU history / 8 GiB history budget. "
        "This port has not reproduced the paper's multi-GPU benchmark."
    )

    @classmethod
    def INPUT_TYPES(cls):
        schema = super().INPUT_TYPES()
        required = schema["required"]
        required.pop("sink")
        required.pop("window")
        required["kv_cache_storage"] = (["cpu", "cpu_pinned"], {"default": "cpu"})
        routers = [name.replace("\\", "/") for name in folder_paths.get_filename_list(ROUTER_FOLDER)
                   if Path(folder_paths.get_full_path_or_raise(ROUTER_FOLDER, name)).with_name("config.json").is_file()]
        required["router_name"] = (routers, {"default": "t2v/model.safetensors"})
        required["far_memory_chunks"] = ("INT", {"default": 1, "min": 1, "max": 2})
        required["history_ram_gib"] = ("FLOAT", {"default": 8.0, "min": 1.0, "max": 256.0,
                                               "tooltip": "Maximum host memory for the history bank and router, separate from model weights."})
        required["memory_mode"] = (["mosaichunk", "matched_window"], {"default": "mosaichunk",
                                    "tooltip": "matched_window keeps the same visual-token budget without historical retrieval, for A/B testing."})
        return schema

    def sample(self, model, positive, latent, video_vae, audio_vae, seed,
               router_name, far_memory_chunks=1, history_ram_gib=8.0, memory_mode="mosaichunk",
               steps=4, video_shift=12.0, audio_shift=3.0, kv_cache_storage="cpu", unique_id=None):
        # Retain saved graphs using the old dedicated sampler. New graphs can
        # use the controller with normal ComfyUI sampler nodes instead.
        from .controller import KEY, strip_controller
        from .conditioning import prepare_model
        from comfy.patcher_extension import WrappersMP
        controllers = model.wrappers.get(WrappersMP.OUTER_SAMPLE, {}).get(KEY, [])
        if controllers:
            controller = controllers[0]
            model = prepare_model(strip_controller(model), controller.conditioning_path,
                                  controller.adapter_path, controller.controls)
        controls = model.model_options.get("mosaichunk_controls")
        if controls is not None:
            video_shift, audio_shift = controls["video_shift"], controls["audio_shift"]
            kv_cache_storage = controls["kv_cache_storage"]
            far_memory_chunks = controls["far_memory_chunks"]
            history_ram_gib = controls["history_ram_gib"]
            memory_mode = controls["memory_mode"]
        if memory_mode not in ("mosaichunk", "matched_window"):
            raise ValueError("memory_mode must be mosaichunk or matched_window.")
        if int(far_memory_chunks) not in (1, 2):
            raise ValueError("far_memory_chunks must be 1 or 2.")
        if kv_cache_storage not in ("cpu", "cpu_pinned"):
            raise ValueError("MosaiChunk history storage must be cpu or cpu_pinned.")
        factory = build_cache_factory(model, router_name, dict(memory_mode=memory_mode,
            far_memory_chunks=far_memory_chunks, history_ram_gib=history_ram_gib))

        return super().sample(
            model=model, positive=positive, latent=latent, video_vae=video_vae, audio_vae=audio_vae,
            seed=seed, steps=steps, video_shift=video_shift, audio_shift=audio_shift,
            sink=1, window=2 + int(far_memory_chunks),
            kv_cache_storage=kv_cache_storage, unique_id=unique_id, cache_factory=factory,
        )


def router_names():
    return [name.replace("\\", "/") for name in folder_paths.get_filename_list(ROUTER_FOLDER)
            if Path(folder_paths.get_full_path_or_raise(ROUTER_FOLDER, name)).with_name("config.json").is_file()]


def build_cache_factory(model, router_name, controls):
    factory = None
    if controls["memory_mode"] == "mosaichunk":
        path = Path(folder_paths.get_full_path_or_raise(ROUTER_FOLDER, router_name))
        config = json.loads(path.with_name("config.json").read_text(encoding="utf-8"))
        if config.get("task") != "text-to-video" or config.get("backbone") != "RAVEN-adapted MiniMax-H3 (H3-AR)":
            raise ValueError("Select the MosaiChunk t2v router, not the LingBot i2v router.")
        args = config["selector"]
        with torch.device("meta"):
            selector = Selector(d_in=args["d_in"], n_views=args["n_views"], d=args["d"], centre=args["centre"])
        selector.load_state_dict(comfy.utils.load_torch_file(str(path), safe_load=True), strict=True, assign=True)
        settings = MemorySettings(
            far_budget=int(controls["far_memory_chunks"]), history_ram_gib=float(controls["history_ram_gib"]),
            sections=config["sections"]["n_sections"],
            descriptor_layers=tuple(config["sections"]["descriptor_layers"]),
            redundancy=config["inference"]["mmr_lam"],
        )
        inv_freq = model.get_model_object("diffusion_model").rope.inv_freq.detach().float().cpu().clone()

        def factory(num_layers, layout, config):
            return MosaiChunkKVCache(num_layers, layout=layout, selector=selector, inv_freq=inv_freq,
                                    settings=settings, storage=config.kv_cache_storage)

    return factory



NODE_CLASS_MAPPINGS = {
    "MosaiChunkH3ModelLoader": MosaiChunkH3ModelLoader,
    "MosaiChunkH3StreamingSampler": MosaiChunkH3StreamingSampler,
    "MosaiChunkH3Conditioning": MosaiChunkH3Conditioning,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MosaiChunkH3ModelLoader": "MosaiChunk H3 RAVEN Model Loader",
    "MosaiChunkH3StreamingSampler": "MosaiChunk H3 Streaming Sampler",
    "MosaiChunkH3Conditioning": "MosaiChunk H3 Conditioning",
}
