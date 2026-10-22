/**
 * JZL 节点外观 — 复刻 XB 的节点配色
 * ======================================
 * 所有 JZL_ 开头的节点：标题栏青蓝色 #16727c，主体紫色 #4F0074。
 * 与 XB_ToolBox 的 Fill-Nodes.appearance 保持一致。
 */

import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "JZL.Appearance",
    async nodeCreated(node) {
        if (node.comfyClass && node.comfyClass.startsWith("JZL_")) {
            node.color = "#16727c";
            node.bgcolor = "#4F0074";
        }
    }
});
