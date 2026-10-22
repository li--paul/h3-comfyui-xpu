"""把本地编译的 llama-server 产物接入 llama-server 后端。

用法:
    python adopt_local_build.py <编译产物目录> [--backend rocm|vulkan|cpu]

示例:
    python adopt_local_build.py D:\\llama.cpp\\bin --backend rocm

脚本会:
    1. 把 llama-server.exe 及全部 dll 复制到 runtime/windows-x64/<tag>/b10436/
    2. 运行 --list-devices 验证
    3. 更新 runtime_config.json 指向新产物
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

NODE_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = NODE_ROOT / "runtime_config.json"


def _find_hip_bin() -> Path | None:
    """定位 HIP SDK 的 bin 目录（含 hipblas.dll / amdhip64*.dll）。"""
    import os

    bins = []
    for env_var in ("HIP_PATH", "ROCM_PATH"):
        root = os.environ.get(env_var)
        if root:
            bins.append(Path(root) / "bin")
    rocm_root = Path("C:/Program Files/AMD/ROCm")
    if rocm_root.is_dir():
        for sub in sorted(rocm_root.iterdir()):
            if sub.is_dir():
                bins.append(sub / "bin")
    for b in bins:
        if b.is_dir():
            names = {d.name.casefold() for d in b.glob("*.dll")}
            if "hipblas.dll" in names and any(n.startswith("amdhip64") for n in names):
                return b
    return None


def _find_rocblas_library() -> Path | None:
    """定位 HIP SDK 的 rocblas library 目录（含 TensileLibrary*.dat）。"""
    import os

    bins = []
    for env_var in ("HIP_PATH", "ROCM_PATH"):
        root = os.environ.get(env_var)
        if root:
            bins.append(Path(root) / "bin" / "rocblas" / "library")
    rocm_root = Path("C:/Program Files/AMD/ROCm")
    if rocm_root.is_dir():
        for sub in sorted(rocm_root.iterdir()):
            if sub.is_dir():
                bins.append(sub / "bin" / "rocblas" / "library")
    for b in bins:
        if b.is_dir() and any(b.glob("*.dat")):
            return b
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="接入本地编译的 llama-server 产物")
    parser.add_argument("src_dir", help="编译产物目录（含 llama-server.exe 和 dll）")
    parser.add_argument("--backend", default="rocm", help="后端标识: rocm/vulkan/cpu (默认 rocm)")
    parser.add_argument("--tag", default="local", help="runtime 子目录名 (默认 local)")
    parser.add_argument("--device", type=int, default=None, help="GPU 设备索引 (HIP 后端指定用哪块显卡)")
    args = parser.parse_args()

    src = Path(args.src_dir).resolve()
    if not src.is_dir():
        print(f"[ERROR] 目录不存在: {src}")
        return 1

    exe_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    exe = src / exe_name
    if not exe.is_file():
        print(f"[ERROR] 未找到 {exe_name}: {src}")
        return 1

    dst = NODE_ROOT / "runtime" / "windows-x64" / args.tag / "b10436"
    dst.mkdir(parents=True, exist_ok=True)

    copied = []
    for f in sorted(src.iterdir()):
        if f.is_file() and f.suffix.lower() in (".exe", ".dll", ".so", ".dylib"):
            shutil.copy2(f, dst / f.name)
            copied.append(f.name)

    if not copied:
        print("[ERROR] 未复制到任何文件")
        return 1
    print(f"[OK] 已复制 {len(copied)} 个文件到 {dst}")

    # 复制 HIP SDK 运行时 dll（llama.cpp HIP 版依赖 hipblas/amdhip64 等）
    hip_bin = _find_hip_bin()
    if hip_bin is not None:
        hip_copied = []
        for f in sorted(hip_bin.glob("*.dll")):
            target = dst / f.name
            if not target.exists():
                shutil.copy2(f, target)
                hip_copied.append(f.name)
        if hip_copied:
            total_mb = sum((dst / n).stat().st_size for n in hip_copied) / (1024 * 1024)
            print(f"[OK] 已复制 HIP SDK 运行时 {len(hip_copied)} 个 dll (约 {total_mb:.0f} MB)")
        else:
            print("[INFO] HIP dll 已存在，跳过复制")
    else:
        print("[WARN] 未检测到 HIP SDK (HIP_PATH/ROCM_PATH)，若缺 hipblas.dll 请手动复制")

    # 复制 rocBLAS Tensile 数据（HIP 推理必需，否则报 TensileLibrary.dat 缺失）
    rocblas_lib = _find_rocblas_library()
    if rocblas_lib is not None:
        rocblas_dst = dst / "rocblas" / "library"
        rocblas_dst.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in sorted(rocblas_lib.glob("*")):
            if f.is_file():
                target = rocblas_dst / f.name
                if not target.exists():
                    shutil.copy2(f, target)
                    n += 1
        if n:
            print(f"[OK] 已复制 rocBLAS Tensile 数据 {n} 个文件到 rocblas/library/")
        else:
            print("[INFO] rocBLAS 数据已存在，跳过复制")
    else:
        print("[WARN] 未找到 rocblas library，HIP 推理可能报 TensileLibrary.dat 缺失")

    # 验证设备
    print(f"[INFO] 运行 {exe_name} --list-devices ...")
    proc = subprocess.run(
        [str(dst / exe_name), "--list-devices"],
        cwd=dst,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    devices_output = (proc.stdout or "") + (proc.stderr or "")
    print(devices_output)
    if proc.returncode != 0:
        print("[WARN] --list-devices 返回非零，请确认编译时启用了正确后端（如 -DGGML_HIP=ON）")

    # 更新 runtime_config.json
    config = {}
    if CONFIG_PATH.is_file():
        try:
            config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            config = {}

    config["schema_version"] = 1
    config["platform"] = "windows"
    config["architecture"] = "x64"
    config["backend"] = args.backend
    config["backend_description"] = f"locally built llama-server ({args.backend})"
    config["executable"] = (dst / exe_name).relative_to(NODE_ROOT).as_posix()
    config["library_dirs"] = [dst.relative_to(NODE_ROOT).as_posix()]
    config.setdefault("runtime_options", {})
    config["runtime_options"].setdefault("n_gpu_layers", "auto")
    config["runtime_options"].setdefault("fit", True)
    config["runtime_options"].setdefault("fit_target_mib", 1536)
    config["runtime_options"].setdefault("flash_attention", "auto")
    if args.device is not None:
        config["runtime_options"]["device"] = args.device
    config["source"] = "local build"

    temporary = CONFIG_PATH.with_name(f"{CONFIG_PATH.name}.tmp")
    temporary.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    temporary.replace(CONFIG_PATH)
    print(f"[OK] Runtime configuration written: {CONFIG_PATH}")
    print("[OK] 完成！重启 ComfyUI 后，把加载器 backend 切到 llama-server 即可使用。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
