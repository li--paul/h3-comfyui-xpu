/**
 * JZL MiniMax 视频调度器 — ComfyUI 前端扩展
 * ==========================================
 * 动态视频端口（IMAGE）+ 配对音频端口（AUDIO）：
 *   - 初始 1 个空视频槽 "*"
 *   - 接入视频后：槽名改为上游节点名，自动在其后生成「上游名（音频）」音频槽
 *   - 最后一个视频槽已连接 → 新增空视频槽
 *   - 视频断开 → 对应音频槽自动移除
 */

import { app } from "../../scripts/app.js";

const NODE_TYPE = "JZL_MiniMax_VideoDispatcher";

// ── helpers ────────────────────────────────────────────────────────────

function getUpstreamNode(self, inputIndex) {
    const graph = self.graph;
    if (!graph) return null;
    let node = self;
    let slot = inputIndex;
    const seen = new Set();
    for (let i = 0; i < 20; i++) {
        const inp = node.inputs?.[slot];
        if (inp?.link == null) return null;
        const link = graph.links?.[inp.link];
        if (!link) return null;
        const src = graph.getNodeById?.(link.origin_id);
        if (!src) return null;
        if (src.type?.includes?.("Reroute") && !seen.has(src.id)) {
            seen.add(src.id);
            node = src;
            slot = 0;
            continue;
        }
        return src;
    }
    return null;
}

// ── extension ──────────────────────────────────────────────────────────

app.registerExtension({
    name: "JZL.MiniMaxVideoDispatcher",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_TYPE) return;
        const orig = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = orig?.apply(this, arguments);
            setTimeout(() => patchInstance(this), 50);
            return r;
        };
    },

    async loadedGraphNode(node) {
        if (node.type === NODE_TYPE) setTimeout(() => patchInstance(node), 200);
    },
});

// ── instance patching ──────────────────────────────────────────────────

function patchInstance(self) {
    if (self.__jzl_vd) return;
    self.__jzl_vd = true;

    // 确保至少一个空视频 IMAGE 槽
    const hasVideo = (self.inputs || []).some(inp => inp.type === "IMAGE");
    if (!hasVideo) self.addInput("*", "IMAGE");

    self.stabilizeInputsOutputs = function () {
        const inputs = this.inputs || [];

        // ── 1. 为每个已连接视频槽生成唯一名（同名加 _N 后缀） ──
        const vIndices = [];
        for (let i = 0; i < inputs.length; i++) {
            if (inputs[i].type === "IMAGE") vIndices.push(i);
        }
        if (vIndices.length === 0) {
            this.addInput("*", "IMAGE");
            return true;
        }

        const nameCount = {};
        const desiredNames = [];  // 与 vIndices 一一对应，空槽为 null
        for (const vi of vIndices) {
            const inp = inputs[vi];
            if (inp.link == null) {
                desiredNames.push(null);
                continue;
            }
            const src = getUpstreamNode(this, vi);
            const base = src?.title || "视频";
            const cnt = nameCount[base] || 0;
            nameCount[base] = cnt + 1;
            desiredNames.push(cnt === 0 ? base : base + "_" + cnt);
        }

        let ch = false;

        // 应用视频槽名
        for (let k = 0; k < vIndices.length; k++) {
            const inp = inputs[vIndices[k]];
            const nm = desiredNames[k];
            if (nm !== null && inp.name !== nm) {
                inp.name = nm;
                ch = true;
            }
        }

        // ── 2. 配对音频槽（每个视频槽一个独立音频槽） ──
        const validAudios = new Set();
        for (const nm of desiredNames) {
            if (nm !== null) validAudios.add(nm + "（音频）");
        }
        // 清理孤立音频槽（对应视频已断开）
        for (let i = inputs.length - 1; i >= 0; i--) {
            const inp = inputs[i];
            if (inp.name && inp.name.includes("（音频）")) {
                if (!validAudios.has(inp.name)) {
                    this.removeInput(i);
                    ch = true;
                }
            }
        }
        // 添加缺失音频槽
        for (const nm of desiredNames) {
            if (nm === null) continue;
            const audioName = nm + "（音频）";
            if (!inputs.some(x => x.name === audioName)) {
                this.addInput(audioName, "AUDIO");
                ch = true;
            }
        }

        // ── 3. 最后一个视频槽已连接 → 新增空视频槽 ──
        const lastV = inputs[vIndices[vIndices.length - 1]];
        if (lastV?.link != null) {
            this.addInput("*", "IMAGE");
            ch = true;
        }

        // ── 4. 移除多余空视频槽（保留最后一个空的） ──
        const vIdx2 = [];
        for (let i = 0; i < inputs.length; i++) {
            if (inputs[i].type === "IMAGE") vIdx2.push(i);
        }
        for (let di = vIdx2.length - 1; di >= 0; di--) {
            const inp = inputs[vIdx2[di]];
            if (inp.link == null && di < vIdx2.length - 1) {
                this.removeInput(vIdx2[di]);
                ch = true;
            }
        }

        return ch;
    };

    const origConn = self.onConnectionsChange;
    self.onConnectionsChange = function (type, index, slot, connected, link_info, ...rest) {
        origConn?.apply?.(this, [type, index, slot, connected, link_info, ...rest]);
        this.scheduleStabilize?.();
    };

    self.scheduleStabilize = function (ms = 80) {
        if (this._jzlVdSched) return;
        this._jzlVdSched = true;
        if (this._jzlVdTimer) clearTimeout(this._jzlVdTimer);
        this._jzlVdTimer = setTimeout(() => {
            this._jzlVdSched = false;
            this._jzlVdTimer = null;
            this.stabilizeInputsOutputs?.();
            this.graph?.setDirtyCanvas?.(true, true);
        }, ms);
    };
}
