"""JZL MiniMax H3 — 参考总线（打包/解包）节点。

48 个固定素材端口打包成一条总线（dict）传输，另一端解包还原。
接口：8角色 + 8场景 + 8道具 + 8音频 + 8视频 + 8视频（音频）。
只做传输，不参与调度。
"""

# 端口顺序固定，out/in 两端一致
BUS_KEYS = tuple(
    [f"角色{ch}" for ch in "ABCDEFGH"]
    + [f"场景{ch}" for ch in "ABCDEFGH"]
    + [f"道具{ch}" for ch in "ABCDEFGH"]
    + [f"音频{ch}" for ch in "ABCDEFGH"]
    + [f"视频{ch}" for ch in "ABCDEFGH"]
    + [f"视频{ch}（音频）" for ch in "ABCDEFGH"]
)

# 自定义总线类型：out/in 只能互连，防接错
BUS_TYPE = "JZL_REF_BUS"


def _bus_type(key):
    """按端口名推断类型：含「音频」为 AUDIO，其余为 IMAGE。"""
    return "AUDIO" if "音频" in key else "IMAGE"


class JZL_MiniMaxH3RefBusOut:
    """参考总线（打包）：48 个素材端口 → 1 条总线。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {key: (_bus_type(key),) for key in BUS_KEYS},
        }

    RETURN_TYPES = (BUS_TYPE,)
    RETURN_NAMES = ("参考总线",)
    FUNCTION = "pack"
    CATEGORY = "JZL/MiniMax"

    def pack(self, **kwargs):
        bus = {key: kwargs.get(key) for key in BUS_KEYS}
        return (bus,)


class JZL_MiniMaxH3RefBusIn:
    """参考总线（解包）：1 条总线 → 48 个素材端口。"""

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
