#!/usr/bin/env python
"""vLLM TP 压测 + PCIe 链路/引擎监控
向 vLLM API 发并发请求(触发 TP allreduce/P2P 通信), 同时监控:
  1. PCIe 链路速率 (数据面 max vs current, 速率<32GT/s 判降速)
  2. 吞吐 (tokens/s) 曲线
用法: python stress_vllm.py [并发] [每请求max_tokens]
"""
import json
import subprocess
import sys
import threading
import time
import urllib.request

URL = "http://localhost:8000/v1/completions"
PROMPT = "Explain the theory of relativity in detail." * 5
CONCURRENCY = int(sys.argv[1]) if len(sys.argv) > 1 else 8
MAX_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 256
DURATION = int(sys.argv[3]) if len(sys.argv) > 3 else 180

# 数据面链路 (速率<32GT/s 或 宽度<设计值 即降速)
LINK_PATHS = [
    ("0a:01.0", "rootport", 32.0, 16),
    ("0b:00.0", "switch-up", 32.0, 16),
    ("0c:00.0", "sw-down0", 32.0, 8),
    ("0c:01.0", "sw-down1", 32.0, 8),
    ("0c:02.0", "sw-down2", 32.0, 8),
    ("0c:03.0", "sw-down3", 32.0, 8),
]


def read_link(bdf):
    try:
        s = open(f"/sys/bus/pci/devices/0000:{bdf}/current_link_speed").read().strip()
        w = open(f"/sys/bus/pci/devices/0000:{bdf}/current_link_width").read().strip()
        return s, w
    except Exception:
        return "?", "?"


def check_links():
    bad = []
    for bdf, tag, dspeed, dwidth in LINK_PATHS:
        s, w = read_link(bdf)
        try:
            speed = float(s.split()[0])
            width = int(w)
        except (ValueError, IndexError):
            continue
        if speed < dspeed or width < dwidth:
            bad.append(f"{bdf}({tag}) {s}x{w} < design {dspeed}x{dwidth}")
    return bad


def request_once(idx):
    body = json.dumps({
        "model": "/home/lm/work/data/Qwen3.6-27B",
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
    }).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    try:
        t0 = time.time()
        r = urllib.request.urlopen(req, timeout=300)
        data = json.loads(r.read())
        dt = time.time() - t0
        n = len(data.get("choices", [{}])[0].get("text", "").split())
        return n, dt
    except Exception as e:
        return 0, 0


# ---- 吞吐统计 ----
lock = threading.Lock()
stats = {"ok": 0, "tokens": 0, "secs": 0.0, "err": 0}
done = [False]


def worker(idx):
    while not done[0]:
        n, dt = request_once(idx)
        with lock:
            if n > 0:
                stats["ok"] += 1
                stats["tokens"] += n
                stats["secs"] += dt
            else:
                stats["err"] += 1


print(f"=== 压测: concurrency={CONCURRENCY} max_tokens={MAX_TOKENS} duration={DURATION}s ===")
print(f"初始链路: {'DOWNGRADE '+str(check_links()) if check_links() else 'OK'}")

threads = [threading.Thread(target=worker, args=(i,)) for i in range(CONCURRENCY)]
for t in threads:
    t.start()

last_ok, last_tok, last_t = 0, 0, time.time()
t_end = time.time() + DURATION
downgrades = 0
while time.time() < t_end:
    time.sleep(5)
    with lock:
        cur_ok, cur_tok, cur_secs, cur_err = stats["ok"], stats["tokens"], stats["secs"], stats["err"]
    dt = time.time() - last_t
    d_ok, d_tok = cur_ok - last_ok, cur_tok - last_tok
    tps = d_tok / dt if dt > 0 else 0
    bad = check_links()
    if bad:
        downgrades += 1
    print(f"[{int(time.time()-t_end+DURATION)}s] ok={cur_ok} tok={cur_tok} "
          f"rate={tps:.1f} tok/s err={cur_err} link={'DOWNGRADE '+str(bad) if bad else 'OK'}")
    last_ok, last_tok, last_t = cur_ok, cur_tok, time.time()

done[0] = True
for t in threads:
    t.join()

with lock:
    print(f"\n=== 结果: ok={stats['ok']} tokens={stats['tokens']} "
          f"avg_tokens_req={stats['tokens']/max(stats['ok'],1):.0f} "
          f"total_time={stats['secs']:.0f}s errors={stats['err']} ===")
print(f"链路降速检测次数: {downgrades}")