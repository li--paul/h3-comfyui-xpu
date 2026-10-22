"""JZL MiniMax H3 — ref2va 参考总线（打包/解包）节点。

对接「MiniMax H3 Reference to Video」节点（固定接口版）：
- ref_image_0 ~ ref_image_8  ：9 张参考图
- ref_video_0 ~ ref_video_2  ：3 段参考视频（IMAGE 帧序列）
- ref_video_audio_0 ~ ref_video_audio_2：3 段视频同步音轨
- ref_audio_0 ~ ref_audio_2   ：3 段独立音频

共 18 个固定接口，打包成一条总线（dict）传输，另一端解包还原。
接口名与「MiniMax H3 Reference to Video」的输入接口名完全对齐，
解包端可直接连线对接。
"""

# 端口顺序固定，out/in 两端一致
BUS_KEYS = tuple(
    [f"ref_image_{i}" for i in range(9)]
    + [f"ref_video_{i}" for i in range(3)]
    + [f"ref_video_audio_{i}" for i in range(3)]
    + [f"ref_audio_{i}" for i in range(3)]
)

# 自定义总线类型：out/in 只能互连，防接错（与 JZL_REF_BUS 区分）
BUS_TYPE = "JZL_REF2VA_BUS"


def _bus_type(key):
    """按端口名推断类型：含 audio 为 AUDIO，其余为 IMAGE。"""
    return "AUDIO" if "audio" in key else "IMAGE"


class JZL_MiniMaxH3Ref2vaBusOut:
    """ref2va 参考总线（打包）：18 个固定接口 → 1 条总线。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {key: (_bus_type(key),) for key in BUS_KEYS},
        }

    RETURN_TYPES = (BUS_TYPE,)
    RETURN_NAMES = ("ref2va参考总线",)
    FUNCTION = "pack"
    CATEGORY = "JZL/MiniMax"

    def pack(self, **kwargs):
        bus = {key: kwargs.get(key) for key in BUS_KEYS}
        return (bus,)


class JZL_MiniMaxH3Ref2vaBusIn:
    """ref2va 参考总线（解包）：1 条总线 → 18 个固定接口，对接 Reference to Video。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"bus": (BUS_TYPE, {"force_input": True})},
        }

    RETURN_TYPES = tuple(_bus_type(key) for key in BUS_KEYS)
    RETURN_NAMES = BUS_KEYS
    FUNCTION = "unpack"
    CATEGORY = "JZL/MiniMax"

    def unpack(self, bus):
        if not isinstance(bus, dict):
            bus = {}
        return tuple(bus.get(key) for key in BUS_KEYS)
