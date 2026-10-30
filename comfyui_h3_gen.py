#!/usr/bin/env python3
"""Build and submit a MiniMax-H3 video job to a running ComfyUI (native H3 nodes).

Two modes, both wired the way the image's own templates wire them:

  t2v    UNETLoader(fl2va) + fl2v turbo LoRA -> MiniMaxH3ImageToVideo (no first_frame)
  ref2v  UNETLoader(ref2va) + ref2v turbo LoRA -> MiniMaxH3ReferenceToVideo
         with one `ref_images.ref_image_N` autogrow slot per reference image

Both run at 1376x768 / 124 frames / 24 fps (5.17 s). `--sol` adds a
BlockSparseAttention(sol-attn) patch, which is worth ~1.25x per step.

Reference images must already be visible to the server. `gen_video.sh` uploads them
through POST /upload/image, which is why the server's --input-directory does not have
to change per project (leaving it out is what silently drops the reference rows:
token count 41042 -> 38970 and frame mean 109 -> 87).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

BASE_URL = os.environ.get("COMFY_URL", "http://127.0.0.1:8188")

MODELS = {
    "t2v": {
        "dit": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "lora": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
    },
    "ref2v": {
        "dit": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "lora": "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors",
    },
}
CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(
        BASE_URL + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=60).read())


def get(path: str) -> dict:
    return json.loads(urllib.request.urlopen(BASE_URL + path, timeout=60).read())


def build_graph(args, ref_names: list[str]) -> dict:
    spec = MODELS[args.mode]
    g: dict[str, dict] = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": spec["dit"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": CLIP, "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
        "6": {"class_type": "LoraLoaderModelOnly",
              "inputs": {"model": ["1", 0], "lora_name": spec["lora"],
                         "strength_model": args.lora_strength}},
    }

    if args.mode == "ref2v":
        g["7"] = {"class_type": "MiniMaxH3ReferenceToVideo",
                  "inputs": {"clip": ["2", 0], "vae": ["3", 0], "audio_vae": ["4", 0],
                             "prompt": args.prompt, "width": args.width,
                             "height": args.height, "length": args.length,
                             "ref_image_size": "match"}}
        for i, name in enumerate(ref_names):
            node = f"5{i}"
            g[node] = {"class_type": "LoadImage", "inputs": {"image": name}}
            g["7"]["inputs"][f"ref_images.ref_image_{i}"] = [node, 0]
    else:
        g["7"] = {"class_type": "MiniMaxH3ImageToVideo",
                  "inputs": {"clip": ["2", 0], "vae": ["3", 0],
                             "prompt": args.prompt, "width": args.width,
                             "height": args.height, "length": args.length}}

    model_src: list = ["6", 0]
    if args.sol:
        g["20"] = {"class_type": "BlockSparseAttention",
                   "inputs": {"model": ["6", 0], "selection": "sol-attn",
                              "selection.tau": args.tau,
                              "start_percent": args.sol_start,
                              "end_percent": 1.0, "dense_blocks": "",
                              "min_tokens": 12288, "extra_tokens": 256,
                              "sink_conditioning": "exact_kv_and_rows",
                              "verbose": False}}
        model_src = ["20", 0]

    g.update({
        "8": {"class_type": "BasicScheduler",
              "inputs": {"model": model_src, "scheduler": args.scheduler,
                         "steps": args.steps, "denoise": 1.0}},
        "9": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": args.sampler}},
        "10": {"class_type": "BasicGuider",
               "inputs": {"model": model_src, "conditioning": ["7", 0]}},
        "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": args.seed}},
        "12": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["11", 0], "guider": ["10", 0], "sampler": ["9", 0],
                          "sigmas": ["8", 0], "latent_image": ["7", 1]}},
        "13": {"class_type": "VAEDecode", "inputs": {"samples": ["12", 0], "vae": ["3", 0]}},
        "14": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["12", 0], "vae": ["4", 0]}},
        "15": {"class_type": "CreateVideo",
               "inputs": {"images": ["13", 0], "audio": ["14", 0], "fps": float(args.fps)}},
        "16": {"class_type": "SaveVideo",
               "inputs": {"video": ["15", 0], "filename_prefix": args.prefix,
                          "format": "auto"}},
    })
    return g


def main() -> int:
    global BASE_URL
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["t2v", "ref2v"], required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--ref-image", action="append", default=[],
                    help="filename already visible to the server (gen_video.sh uploads them)")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--sampler", default="res_multistep")
    ap.add_argument("--scheduler", default="simple")
    ap.add_argument("--width", type=int, default=1376)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--length", type=int, default=124)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--lora-strength", type=float, default=1.0)
    ap.add_argument("--sol", action="store_true")
    ap.add_argument("--tau", type=float, default=1.3)
    ap.add_argument("--sol-start", type=float, default=0.0)
    ap.add_argument("--out-dir", default="",
                    help="download the result here via /view (the server's own output "
                         "directory is whatever it was started with, so reusing a server "
                         "would otherwise scatter artefacts)")
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--base-url", default=BASE_URL)
    args = ap.parse_args()
    BASE_URL = args.base_url

    with open(args.prompt_file, encoding="utf-8") as fh:
        args.prompt = fh.read().strip()
    if not args.prompt:
        print("error: prompt file is empty", file=sys.stderr)
        return 2
    if args.mode == "ref2v" and not args.ref_image:
        print("error: ref2v needs at least one --ref-image", file=sys.stderr)
        return 2
    if len(args.ref_image) > 9:
        print("error: H3 accepts at most 9 reference images", file=sys.stderr)
        return 2

    graph = build_graph(args, args.ref_image)
    print(f"mode={args.mode} nodes={len(graph)} refs={len(args.ref_image)} "
          f"steps={args.steps} {args.sampler}/{args.scheduler} "
          f"{args.width}x{args.height}/{args.length}f seed={args.seed} sol={args.sol}",
          flush=True)

    t0 = time.time()
    pid = post("/prompt", {"prompt": graph, "client_id": "h3-gen"})["prompt_id"]
    print(f"queued {pid}", flush=True)

    while time.time() - t0 < args.timeout:
        time.sleep(10)
        hist = get(f"/history/{pid}")
        if pid in hist:
            status = hist[pid].get("status", {})
            print(f"done in {time.time() - t0:.1f}s status={status.get('status_str')}", flush=True)
            if status.get("status_str") != "success":
                print(json.dumps(status, ensure_ascii=False)[:2000], file=sys.stderr)
                return 1
            saved = 0
            for out in (hist[pid].get("outputs") or {}).values():
                for items in out.values():
                    for item in items if isinstance(items, list) else []:
                        if not (isinstance(item, dict) and item.get("filename")):
                            continue
                        name, sub = item["filename"], item.get("subfolder", "")
                        print(f"output {sub}/{name}", flush=True)
                        if not args.out_dir:
                            saved += 1
                            continue
                        # Pull the artefact through /view so it lands in the project
                        # regardless of which --output-directory the server was started with.
                        from urllib.parse import quote
                        url = (f"{BASE_URL}/view?filename={quote(name)}"
                               f"&subfolder={quote(sub)}&type=output")
                        dest = os.path.join(args.out_dir, name)
                        if os.path.exists(dest) and os.path.getsize(dest) > 0:
                            # SaveVideo already wrote it there (the server was started
                            # with --output-directory pointing at this project); do not
                            # fetch a duplicate copy.
                            print(f"already in place {dest} ({os.path.getsize(dest)} bytes)", flush=True)
                            saved += 1
                            continue
                        if os.path.exists(dest):  # never overwrite a previous run
                            stem, ext = os.path.splitext(name)
                            n = 2
                            while os.path.exists(os.path.join(args.out_dir, f"{stem}-{n}{ext}")):
                                n += 1
                            dest = os.path.join(args.out_dir, f"{stem}-{n}{ext}")
                        with urllib.request.urlopen(url, timeout=600) as resp, open(dest, "wb") as fh:
                            while chunk := resp.read(1 << 20):
                                fh.write(chunk)
                        print(f"saved {dest} ({os.path.getsize(dest)} bytes)", flush=True)
                        saved += 1
            if saved == 0:
                print("error: the job reported success but produced no file", file=sys.stderr)
                return 1
            return 0
    print("timeout waiting for the job", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
