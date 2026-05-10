DEHAZE + CLIP Scoring — 使用说明

目标
- 在本仓库中运行图像去雾（dehaze）并使用 CLIP 对每张去雾图像按候选文本进行打分。
- 生成的打分文件便于快速查看每张图像的 top-K 标签（以及 CLIP 给出的 softmax 概率）。

主要文件
- `CoA/Eval.py`  — 主脚本，包含 `dehaze(model, image_path, folder)` 函数和可选 CLIP 打分逻辑。
- `outputs/clip_dehaze/clip_descriptions.txt` — 逐图的 CLIP 打分输出（原始运行产生，可能包含重复或旧格式）。
- `outputs/clip_dehaze/clip_descriptions.cleaned.txt` — 清理后的描述文件（每行最多 top-3，无重复标签），如果已存在则优先查看。
- `outputs/clip_dehaze/clip_prompts_used.txt` — 每张图像写入时记录的候选 prompts 列表。
- `outputs/clip_dehaze/clip_scores_info.txt` — 解释文件，说明分数含义（softmax 概率）。
- `CoA/scripts/clean_clip_descriptions.py` — 将 `clip_descriptions.txt` 清理为 `clip_descriptions.cleaned.txt` 的脚本。

快速准备
1. 确认 Python 环境满足依赖（在 `CoA/requirements.txt` 中）。通常需要：torch, torchvision, pillow, numpy 等。
2. 确认 `CoA/clip_model/` 中存在 CLIP 权重（如 `ViT-B-32.pt` 或仓库自带的加载路径），否则 CLIP 打分会被跳过。
3. 检查 `MODEL_PATH` 在 `CoA/Eval.py` 是否指向有效模型文件（默认指向 `./model/EMA_model/EMA_r.pth`）。

示例：只做去雾（不打分）
- 在不修改 `EVAL_MODE` 的情况下直接运行脚本（默认会在最后的 else 分支运行去雾流程）：

```bash
python3 CoA/Eval.py
```

示例：小批量带 CLIP 打分（测试）
- 我们建议使用脚本调用 `dehaze()` 进行受控小批量测试（不会覆盖主输出文件）：

```bash
# 在仓库根运行以下 python 命令来处理前20张图并把结果写入 outputs/clip_dehaze/test_batch20
python3 - << 'PY'
import runpy, glob, os
E = runpy.run_path('CoA/Eval.py')
imgs = sorted(glob.glob(os.path.join('dataset/Haze4K/test/hazy_school','*jpg')))[:20]
outdir = os.path.join('outputs','clip_dehaze','test_batch20')
os.makedirs(outdir, exist_ok=True)
for im in imgs:
    E['dehaze'](E.get('model'), im, outdir)
PY
```

查看输出
- 主输出目录：`outputs/clip_dehaze/`
  - `clip_descriptions.txt`：按时间追加的原始描述（可能来自多次运行）。
  - `clip_descriptions.cleaned.txt`：清理后文件（每行为 `filename -> [(label, score), ...]`）。
  - `clip_prompts_used.txt`：记录了为每张图像使用的候选 prompts。
  - `clip_scores_info.txt`：解释 scores 的含义（softmax 概率）。

如何解读分数
- 每个 `(label, score)` 中的 score 是 CLIP 在对该图像和这些 candidate 文本做 softmax 后得到的概率值（0-1）。
- 这些分数仅在同一组候选 prompts 内有意义。分数较高表示 CLIP 模型认为该 text 更能描述图像。
- 这不是生成型 caption；它仅在候选文本集合中做相对比较。

常见问题与排查
- 没有生成 `clip_descriptions.txt`：
  - 检查 `CoA/Eval.py` 中 `clip_mod` 与 `clip_model_global` 是否被成功加载（文件 `outputs/clip_dehaze/clip_init_status.txt` 会记录加载情况）。
  - 如果 CLIP 未加载，脚本会跳过打分；确认 `CoA/clip_model/` 中存在权重或网络可从 repo-local CLIP 加载。
- 出现 ValueError / RuntimeError（如 array->scalar 转换）：
  - 这通常由旧的索引/转换代码引起。我们已在 `Eval.py` 中修复常见路径，若仍报错请把最新的 `outputs/clip_dehaze/clip_err.log` 提供给我。
- 输出有重复标签或非常多条目：
  - 这通常因为多次运行叠加到同一 `clip_descriptions.txt` 导致。使用 `CoA/scripts/clean_clip_descriptions.py` 来生成 `clip_descriptions.cleaned.txt` 或在处理前手动备份并清空旧文件。

如何定制候选 prompts（快速示例）
- 候选 prompts 位于 `CoA/Eval.py` 中的 `candidates = [...]` 列表，或者可使用 `CoA/CLIP/prompts.py` / `CoA/CLIP/prompt2.py` 来生成更丰富或随机化的 candidate 列表。
- 注意：`prompts.py` 与 `prompt2.py` 当前会在导入时打印/随机化，建议将那两个文件重构为无副作用的生成函数后再调用（或使用我为你准备的清理脚本）。

如何做一次干净的重跑（推荐）
1. 备份旧文件：
   mv outputs/clip_dehaze/clip_descriptions.txt outputs/clip_dehaze/clip_descriptions.bak.txt
2. 清理目录并重跑（示例：小批量）
   - 见上面“小批量带 CLIP 打分”的命令。


