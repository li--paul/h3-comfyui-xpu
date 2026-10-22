# 发布 / 同步规范

本文件规定 **ComfyUI-JZL-MiniMax-H3** 仓库哪些文件必须上传、哪些禁止上传，以及发布到 ComfyUI Registry 的流程。

## 一、必须上传（白名单）

| 文件 | 作用 |
|------|------|
| `__init__.py` | V3 注册入口（`comfy_entrypoint`） |
| `nodes.py` | 节点实现 |
| `pyproject.toml` | 包元数据 + `[tool.comfy]` Registry 识别信息 |
| `README.md` | 使用说明 |
| `LICENSE` | MIT 许可（Registry 要求） |
| `.gitignore` | 排除无用文件 |

可选：`icon.png`（节点图标）、`workflows/`（示例工作流 JSON）。

## 二、禁止上传（黑名单）

| 类型 | 示例 | 原因 |
|------|------|------|
| Python 缓存 | `__pycache__/`、`*.pyc`、`*.pyo` | 机器生成，无源码价值 |
| 虚拟环境 | `.venv/`、`venv/` | 体积巨大 |
| 临时验证脚本 | `_check.py`、`_verify*.py`、`test_*.py` | 一次性调试产物 |
| 打包产物 | `*.egg-info/`、`dist/`、`build/` | 构建生成 |
| 系统文件 | `.DS_Store`、`Thumbs.db` | 系统噪音 |
| 日志 | `*.log` | 运行产物 |

**铁律：节点源码是 Python，必须上传；「没用的 Python」指的是缓存、临时脚本、测试文件，不是节点本体。**

## 三、发布到 ComfyUI Registry 流程

1. 确认 `pyproject.toml` 里 `[tool.comfy]` 的 `PublisherId` 唯一（本仓库 = `wjluoxiao`）。
2. `git push` 到 GitHub。
3. 打开 https://registry.comfy.org/publish ，选择本仓库，按提示发布版本。
4. 每次发版前自检（见下）。

## 四、发版前自检清单

```powershell
# 1. 语法校验
python -c "import ast; [ast.parse(open(f, encoding='utf-8').read()) for f in ['__init__.py', 'nodes.py']]; print('OK')"

# 2. 确认无缓存/临时文件进入 git
git status --short

# 3. 确认关键字段
Select-String -Path pyproject.toml -Pattern 'PublisherId|DisplayName|Repository'
```

## 五、版本号规则

- `pyproject.toml` 的 `version` 与 Git tag 保持一致（如 `v0.1.0`）。
- 每次改动节点行为（非文档/注释）都必须递增版本号。
