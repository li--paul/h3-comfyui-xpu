# MiniMax-H3「神笔马良」出片环境 · 环境与模型版本清单

- 采集时间：2026-10-20
- 采集对象：Linux 出片服务器（Intel Arc Pro B60 ×32）
- 采集范围：交付包 `神笔马良_demo` 的运行环境、模型权重与默认推理参数

---

## 1. 硬件

| 项目 | 规格 |
| --- | --- |
| CPU | 2 × Intel Xeon Gold 6530（128 线程，2 路） |
| 内存 | 2 011 GiB |
| GPU | Intel Arc Pro B60 ×32，单卡显存 24 480 MB（23.9 GiB） |
| GPU 计算单元 | 每卡 20 个 Xe 子切片 / 160 个执行单元 |
| 存储 | NVMe 系统与数据分区 7 TB（可用 6.2 TB） |
| PCIe 拓扑 | 32 张卡分 8 组，每组 4 张共用一条上行链路 |

---

## 2. 操作系统与驱动

| 项目 | 版本 |
| --- | --- |
| 操作系统 | Ubuntu 26.04 LTS（x86_64） |
| 内核 | 7.0.0-38-generic（Intel `xe` 图形驱动） |
| GPU 驱动版本 | 1.17.39395+13 |
| 计算运行时 | Intel oneAPI Unified Runtime over Level-Zero V2 |
| 监控工具 | xpu-smi 1.3.7.20260617 |

---

## 3. oneAPI 与编译器

| 组件 | 版本 |
| --- | --- |
| oneAPI 安装路径 | `/opt/intel/oneapi` |
| 当前生效版本 | 2026.1（同时保留 2026.0 目录，`latest` 指向 2026.1） |
| DPC++/C++ 编译器（icpx） | 2026.1.1（构建号 2026.1.1.20260724） |
| oneMKL | 2026.1 |
| oneDNN | 2026.0 |
| oneCCL | 2022.1 |

环境挂载方式：`source /opt/intel/oneapi/setvars.sh`。

---

## 4. Python 运行时与关键依赖

| 组件 | 版本 |
| --- | --- |
| Python | 3.12.13（包内自带 runtime/python，不依赖系统 Python） |
| PyTorch（Intel XPU 版） | 2.14.1+xpu |
| torchvision | 0.29.1+xpu |
| torchaudio | 2.11.0+xpu |
| transformers | 5.19.0 |
| tokenizers | 0.23.2 |
| accelerate | 1.15.0 |
| diffusers | 0.41.0 |
| safetensors | 0.8.0 |
| numpy | 2.5.3 |
| opencv-python-headless | 5.0.0.93 |
| pillow | 12.3.0 |
| imageio-ffmpeg | 0.6.0 |

---

## 5. 推理框架与插件

| 组件 | 版本 / 提交 |
| --- | --- |
| ComfyUI | 0.39.0 |
| comfyui-frontend-package | 1.55.14 |
| comfyui-workflow-templates | 0.11.78 |
| comfy-aimdo（动态显存与权重流式装载） | 0.5.6 |
| comfy-kitchen（算子后端） | 0.2.37 |
| 自定义节点 ComfyUI-JZL-MiniMax-H3（短剧导演台、资产管理、出片节点） | 82acd16 |
| 自定义节点 ComfyUI-MiniMaxH3_LatentUpscaler | 2e568df |

---

## 6. 模型权重清单

权重目录 `models/ComfyUI-MiniMax-H3/`，合计约 53 GB。

| 用途 | 文件 | 大小 | 张量数 | 精度 |
| --- | --- | --- | --- | --- |
| 扩散主模型（参考图生成音视频，REF2VA） | `diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors` | 20.97 GB | 932 | INT8（ConvRot 旋转量化） |
| 文本与多模态编码器 | `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | 27.14 GB | 1 602 | INT8 量化 |
| 视频 VAE | `vae/minimax_h3_video_vae_bf16.safetensors` | 5.21 GB | 562 | BF16 |
| 音频 VAE | `vae/minimax_h3_audio_vae_fp32.safetensors` | 0.61 GB | 917 | FP32 |
| 8 步加速 LoRA | `loras/minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors` | 1.96 GB | 624 | BF16 |

---

## 7. 模型构成与精度说明

- **扩散主模型**：MiniMax-H3 音视频联合扩散模型，Flow Matching 架构（模型自述类型 `FLOW_AV`），同一网络同时去噪视频 latent 与音频 latent，单次推理产出带音轨的成片。
- **量化格式**：INT8 + ConvRot 分组量化。权重以 INT8 驻留显存，计算时按算子反量化执行；后端报告的可执行算子集为 `int8_tensorwise`、`convrot_w4a4`、`w6a8_int8`、`asym_w4a8_int8`、`float8_e4m3fn`、`float8_e5m2`、`mxfp8`、`nvfp4`，均为模拟（emulated）实现。
- **文本与多模态编码器**：Qwen3-VL-32B，50 层，取第 50 层未归一化隐状态作为条件；同时编码文本提示词与参考图/参考视频/参考音频。
- **视频 VAE**：z_channels = 24，时间压缩 4×，clip 长度 17；解码器为 3D 卷积 + ViT 解码（36 层、32 头、RoPE），编码器为因果 3D 卷积。
- **音频 VAE**：FP32，与视频 VAE 独立。
- **加速 LoRA**：8 步出图 LoRA，rank 128、alpha 8，由 Diffusers PEFT 格式转换为 ComfyUI 通用格式，基线模型为 MiniMax-H3。
- **显存精度**：视觉 VAE 以 BF16 运行（ComfyUI `--bf16-vae`）；每次解码前先全量卸载采样模型，显卡同一时刻只驻留一个大模型。

---

## 8. 默认推理参数

| 参数 | 值 |
| --- | --- |
| 分辨率 | 1 376 × 768（16:9） |
| 每段帧数 | 124 帧（约 5 秒） |
| 采样器 / 调度器 | res_multistep / beta |
| 采样步数 | 8（配合 turbo LoRA） |
| CFG | 1.0 |
| 种子 | 固定，第 N 段 = 1000 + N（可完整复现） |
| 分段长度 | 1 ~ 56 段可配 |
| 解码方式 | VAE 解码（原版优化），BF16，解码前卸载显存模型 |
| 输出 | MP4（视频 + 音轨），分段与整片两种产物 |

---

## 9. 运行形态

- **一卡一实例**：每张卡运行一个 ComfyUI 进程，通过 `ZE_AFFINITY_MASK` 绑定单卡，端口 8190 + 序号，各自独立的 user 与 output 目录。
- **权重驻留**：每实例装载文本编码器约 5 GB 与扩散主模型约 15 GB 到显存，其余约 11 GB 随采样步流式换入。
- **并发出片**：32 卡时 32 个实例同时工作，一实例负责一段，段与卡一一对应。
- **产物结构**：`shots/shotNN.mp4`（分段）、`full/<片名>_N段.mp4`（成片）、`logs/`（每实例日志）、`prompts/`（本次提交的 API 请求）、`report.md`（逐段耗时与失败记录）。
- **离线部署**：包内自带 Python 运行时、uv 与全部 wheel，无需联网，虚拟环境缺失时可离线重建。

---

## 10. 性能实测

| 场景 | 每段耗时 | 说明 |
| --- | --- | --- |
| 单卡（Arc Pro B60） | 约 10.8 分钟 | 124 帧、8 步 |
| 16 卡并行 | 13 分 06 秒 ~ 15 分 54 秒 | 每步 78 ~ 84 秒 |
| 32 卡并行 | 每步 70 ~ 85 秒 | 装载阶段约 2 分钟 |

单段耗时构成：权重装载约 2 分钟，8 步采样约 10.5 分钟，VAE 解码约 1 分钟。

---

## 11. 并发装载的已知限制

32 张卡同时把约 20 GB 权重搬入各自显存时，个别卡会在 Level Zero 层出现设备挂起或 `UR_RESULT_ERROR_DEVICE_LOST`（设备掉线）。现场规避方式：装载分波进行，同组 PCIe 链路只允许一张卡同时装载；实例异常时自动重启并把该段重派给空闲卡。

---

## 附：采集命令

```bash
# oneAPI 与编译器
cat /opt/intel/oneapi/latest/release_notes 2>/dev/null; icpx --version
# 关键 Python 依赖
ComfyUI/.venv/bin/python -c "import torch;print(torch.__version__, torch.xpu.is_available())"
ComfyUI/.venv/bin/python -m pip list
# 显卡与驱动
xpu-smi discovery; xpu-smi --version
# 权重清单
ls -l models/ComfyUI-MiniMax-H3/*/
```
