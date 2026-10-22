"""JZL MiniMax H3 — 手写提示词编辑节点（V1 经典 API）

纯文本编辑 + H3 六段提示词/调度指令快捷插入。插入逻辑在 js/h3_prompt_editor.js。
无输入端口，输出 STRING（H3提示词），可直接接到「分段处理中心」的 shot_text 输入，
或「提示词增强」节点的 script_text 输入。
"""

class JZL_MiniMaxH3PromptEditor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt_text": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "手写 MiniMax H3 提示词。点击「➕ 添加元素」可在光标处另起一行插入六段字段/调度指令/分镜等模板",
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("H3提示词",)
    FUNCTION = "build"
    CATEGORY = "JZL/MiniMax"

    def build(self, prompt_text):
        return (prompt_text,)
