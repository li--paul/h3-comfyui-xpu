"""
JZL MiniMax — 列表分发节点
===========================
上游输入「剧本与镜头处理器」的剧本输出，按分段块（[SHOT_START]...[SHOT_END]）切分，
再按选择的调度类型提取每个分段的完整字段（H3提示词/场景调度/视频调度/音频调度），
分发到 N 个输出端口，每个端口输出一份完整内容，不拆行、不叠加。
"""

import re


class JZL_ListDispatcher:
    """列表分发 — 按分段块切分 → 按调度类型提取 → 动态输出。"""

    MAX_OUTPUTS = 99
    INPUT_IS_LIST = True  # 复刻 XB：统一以列表接收上游数据

    _SECTION_MAP = {
        "H3提示词": "H3_PROMPT",
        "场景调度": "SCENE_INSTRUCTION",
        "视频调度": "VIDEO_INSTRUCTION",
        "音频调度": "AUDIO_INSTRUCTION",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "文本输入": ("STRING", {"forceInput": True}),
                "调度类型": (["H3提示词", "场景调度", "视频调度", "音频调度"], {"default": "H3提示词"}),
                "输出数量": ("INT", {"default": 4, "min": 1, "max": cls.MAX_OUTPUTS, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",) * MAX_OUTPUTS
    RETURN_NAMES = tuple(f"分段{i + 1}" for i in range(MAX_OUTPUTS))
    FUNCTION = "dispatch"
    CATEGORY = "JZL/MiniMax"

    @staticmethod
    def _extract_section(block, section):
        """从单个分段块中提取指定字段的完整内容（不拆行）。"""
        marker = f"==={section}==="
        for part in re.split(r'\n(?====)', block):
            part = part.strip()
            if part.startswith(marker):
                return part[len(marker):].lstrip("\n").rstrip()
        return ""

    def dispatch(self, 文本输入, 调度类型, 输出数量):
        # INPUT_IS_LIST=True：所有参数以列表接收（复刻 XB），兼容单值
        count = int(输出数量[0]) if isinstance(输出数量, list) else int(输出数量)
        section_key = 调度类型[0] if isinstance(调度类型, list) else 调度类型
        section = self._SECTION_MAP.get(section_key, "H3_PROMPT")

        # 1. 输入规整：list → 元素列表；标量字符串 → 单元素
        if isinstance(文本输入, str):
            raw_list = [文本输入]
        else:
            raw_list = [str(x) for x in (文本输入 or []) if x is not None]

        # 2. 每个元素尝试按 [SHOT_START] 块拆 + 提取字段；
        #    无标签时（如分段处理中心已提取好的 list）整个元素就是一段，直接分发
        items = []
        for raw in raw_list:
            raw = raw.replace("\\n", "\n").replace("\\r", "\n")
            blocks = re.findall(r'\[SHOT_START\](.*?)\[SHOT_END\]', raw, re.DOTALL)
            if blocks:
                for b in blocks:
                    items.append(self._extract_section(b, section))
            elif raw.strip():
                items.append(raw)

        # 3. 分发到 N 个端口 + 节点显示框
        count = max(1, min(self.MAX_OUTPUTS, count))
        results = []
        displays = []
        for i in range(self.MAX_OUTPUTS):
            val = items[i] if i < count and i < len(items) else ""
            results.append(val)
            if i < count:
                displays.append(val)
        return {"ui": {"displays": displays}, "result": tuple(results)}
