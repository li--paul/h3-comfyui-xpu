# vLLM TP=8 触发 PCIe 链路降速 — 复现记录

> 日期：2026-09-02
> 硬件：32× Intel Arc Pro B60 + 8× PM50100 Switchtec Gen5，双路 Xeon Gold 6530
> 环境：`uv_scaffolding_xpu`（torch 2.12 + vLLM 0.26 + vllm-xpu-kernels，全新 uv sync）
> 模型：Qwen3.6-27B，TP=8（GPU0-7，根端口 0a + 22）

> **备注（2026-10）**：主环境已升级到 torch 2.14 + vLLM 0.32，`run_vllm_tp.sh` 也改用
> `vllm serve` 入口。下面的复现数据仍来自 2.12 / 0.26 时代，结论未在新版本上重新验证。

## 1. 环境搭建

```bash
git clone https://github.com/li--paul/uv_scaffolding_xpu
cd uv_scaffolding_xpu
bash setup.sh   # 自动 clone vendor/vllm(v0.26) + uv sync (编译 vLLM XPU)
```

**关键环境坑（已解决）**：
1. **triton cache 污染**：`~/.triton/cache` 旧 JIT 产物绑定 system libsycl.so.9，与 venv libsycl.so.8 冲突 → segfault。修复：`rm -rf ~/.triton/cache`
2. **icpx 干扰**：PATH 中 oneAPI icpx 让 triton 用 system .9。修复：启动时 PATH 不含 oneAPI bin，让 triton 走 intel-sycl-rt(.8) 分支

启动脚本：`run_vllm_tp.sh`（清 triton cache + 干净 PATH）

## 2. TP=8 推理

```bash
./run_vllm_tp.sh 8 8000
# engine init 48.6s, API on :8000
```

单请求验证：28 tokens / 14.83s，正常。

## 3. 压测与降速监控

```bash
.venv/bin/python -u stress_vllm.py <并发> <max_tokens> <秒数>
# 并发2×128tok×60s 即触发
```

监控项（`stress_vllm.py` 内置）：
- PCIe 数据面链路速率（max vs current，速率<32GT/s 判降速）
- dmesg 引擎错误（Fault/Engine reset/GuC timeout）

## 4. 结果（核心发现）

### ✅ 成功复现 PCIe 链路降速，且无法自动恢复

| 根端口 | 承载 GPU | 当前速率 | 设计速率 |
| --- | --- | --- | --- |
| **0a:01.0** | GPU0-3（TP8 重负载） | **5.0 GT/s** | 32 GT/s |
| 22:01.0 | GPU4-7（TP8 重负载） | 32 GT/s | 32 GT/s |
| 3e/54/83/96/a9/bc:01.0 | 其他 | 32 GT/s | 32 GT/s |

- **只有承载 TP=8 的根端口 `0a:01.0` 降到 5.0 GT/s（Gen1），其余 7 个正常** → 确为负载触发
- 负载停止 30s+、setpci retrain(68.w=0x20)、bus reset 均**无法恢复**（持久卡死）
- 数据面 `0c:00.0-03`（GPU0-3 下游）仍 32 GT/s → 瓶颈在**根端口→交换机上行**
- 压测期间**无引擎错误**（0 条 Fault/Reset）→ 纯 PCIe PHY/LTSSM 层降速，非 GPU 引擎故障
- 内核**无 AER/带宽通知记录** → 静默降速，难以察觉

### 吞吐表现（降速后）
压测吞吐仍 ~30 tok/s（Gen1 x16=4GB/s 对单次推理足够，但**多卡通信带宽骤降 16 倍**，大规模并行时性能会显著劣化）。

## 5. 意义

这是用户所述 **"torchrun/重通信应用 → pcie downgrade → 无法恢复"** 现象的**直接复现**：
- 触发条件：TP=8 级的多卡重通信（allreduce/P2P 密集）
- 根因：PCIe Gen5 根端口在重负载下链路训练降级（LTSSM 失锁→Gen1），且**不自动恢复**
- 影响：承载该根端口的 GPU 上行带宽从 64GB/s 掉到 4GB/s（1/16）
- 恢复：仅 setpci/bus reset 无效，需**断电重启**（完整链路重新训练）

## 6. 环境文件

- `run_vllm_tp.sh` — 正确环境启动 TP 推理
- `stress_vllm.py` — 压测 + 链路/引擎监控
- 环境根目录：`/home/lm/work/paul/test/uv-scaffolding-xpu/`