# VLM-as-Participant：用本地开源 VLM 复现 clustering 可视化用户研究

这是 `V-Means` 仓库中的一个**独立研究模块**：它与 Qt 桌面界面解耦，拥有
自己的依赖、命令行入口、测试、模型注册表和输出目录，可以单独复制到 GPU
服务器运行。runner 面向支持原生视频内容的 OpenAI-compatible Chat
Completions endpoint；当前附带五个正式模型配置，也可以注册更多满足该接口的
视觉语言模型。运行该模块不需要启动 V-Means GUI。

本目录是独立源仓库 `Slian22/vlm_userstudy` 在 commit
`6d62fc2fba430a02e0496fa08f4c2c4fc632bb29` 的集成快照；两份仓库不会自动
同步。后续从源仓库更新时，应显式核对源码、测试和启动脚本，并重新运行本目录
的完整回归测试。

把 SOTA 开源视觉语言模型当作"参与者"，复现 212 名人类参与者的问卷流程，
结果写入同一个 Google Spreadsheet 的 `VLM_Responses` tab，用于人机对比
（open-weight VLM baseline）。Q6 与人类表单一致，全部 13 项照常施测；
VLM 版本预先规定的调整是：开放题要求提供非空的定性回答。

## 协议（与人类研究对齐）

每个模型 × 每个 run 为**一个会话**：依次原生输入四个视频（processor 内部
采样，不手动抽帧），每个视频后答 Q1–Q9；四个视频全部留在上下文中之后，
最后一轮纯文本追问 Q17–Q23（人类表单的 Overall 页同样无视频）。
（例外：GLM-4.6V——vLLM 的 GLM-4V 实现每个 prompt 最多 1 个视频，
runner 在每个新视频轮前把先前视频替换为文字占位（`max_videos_per_prompt: 1`），
模型靠自己此前的作答回忆，等价于人类不能回看视频；CSV 的 input_mode 列
记为 `native_video_1perprompt`。）
每模型 3 个独立 run；采样用各厂商模型卡推荐参数（config 的 `sampling`，
实际 temperature 记录在 CSV 元数据列），每个请求带 `seed=run_id` 保证
可复现且 run 间独立；统一 num_frames=64/视频（例外：InternVL3.5-38B 用
24 帧——它无视频 token 压缩，~260 token/帧，4×64 帧约 66k token 超出其
40960 上下文；24 帧时最重一轮约 36k。CSV 的 num_frames_per_video 列
如实记录）。不用 temperature=0：
贪婪解码会让思考型模型陷入无限复读（Qwen3.5-9B 实测）。

问卷题干与选项**逐字**取自原始 Google Form（有测试比对响应 CSV 表头），
Q6 全部 13 项与 Google Form 逐字逐序对齐，顺序不变；
Q17–Q19 锚点 1=Strongly disagree/5=Strongly agree，Q20 为
1=Not confident at all/5=Very confident。受限解码（guided decoding）把
每题答案钉死在表单选项集内——这是 radio button 的数字等价物，不含任何
内容提示；实际生效的约束语法记录在 `answer_constraint` 元数据列
（json_schema / structured_outputs / none，逐会话自动协商）。开放题
（Q8/Q9/Q21–Q23）要求 1–3 句非空回答；若没有不清楚或无需改进，模型也必须
明确说明并简短解释原因。

## 模型支持与扩展（当前实验定稿五个）

| tag | 权重 | 卡（node03） | 磁盘 | 轴 |
|---|---|---|---|---|
| qwen3vl-8b | Qwen/Qwen3-VL-8B-Instruct | GPU 3 | ~17G | 消费级基准 |
| qwen3.5-9b | Qwen/Qwen3.5-9B | GPU 3 | ~20G | early-fusion 新一代（与 8B 构成架构对照） |
| minicpm-v-4.5 | openbmb/MiniCPM-V-4_5 | GPU 3 | ~18G | 独立家族 + 视频 token 压缩 |
| internvl3.5-38b | OpenGVLab/InternVL3_5-38B | GPU 4,5 | ~76G | 独立家族 + 中档规模 |
| glm-4.6v | zai-org/GLM-4.6V | GPU 4,5,6,7 | ~212G (bf16) | 旗舰档 106B-MoE + 思考型；A800 无原生 FP8，Marlin 回退与 GLM 维度不兼容，故用 bf16 原始权重 |

总磁盘 ~260G。GLM-4.6V 和 Qwen3.5-9B 是思考型模型（serve 脚本分别带
`--reasoning-parser glm45` / `--reasoning-parser qwen3`）。config 的
`MAX_TOKENS=None`：请求不带 token 上限，思考+答案可用满模型上下文窗口
（serve 脚本的 `--max-model-len`）。采样必须用 config 里各模型的厂商
推荐参数——temperature=0 贪婪解码会让思考型模型无限复读直到撑爆窗口
（`empty content (finish_reason=length)`）。

`serve/glm45v.sh` 和 `serve/qwen3vl_235b_fp8.sh` 是历史参考脚本，不在
`config.MODELS` 或本次五模型实验中，不要正式运行。

这五个模型是当前实验的固定比较组，并不是 runner 的接口上限。要接入其他
模型，在 `config.py` 的 `MODELS` 中增加 tag、模型 ID、采样参数和启动脚本，
并让对应服务提供支持原生视频输入的 OpenAI-compatible endpoint；随后先运行
`--dry-run`、回归测试和独立 pilot。模型特有的帧数或单 prompt 视频数量限制也
应写进该 registry，而不是散落在 runner 中。服务端 `vllm` 没有放进本模块的
客户端 `requirements.txt`，应在 GPU 环境中按模型和硬件单独安装。

## 安全与运行边界

`serve/*.sh` 用于受信任 GPU 节点上的可复现实验，不是经过加固的公共推理服务。
不要把 vLLM 的 8000 端口直接暴露到公网或不受信任的共享网络；runner 最适合在
同一节点通过 `localhost` 访问。跨机器运行时，优先使用 SSH tunnel，并同时配置
防火墙、访问控制和传输加密。若所用 vLLM 版本支持，应显式绑定 loopback 地址。

InternVL 与 MiniCPM-V 的脚本需要 `--trust-remote-code`。正式实验前应审查并固定
不可变的 Hugging Face model/code revision，在无额外凭据的专用低权限环境中运行；
不要把 Google service-account key、HF token 或 SSH agent 放进模型服务进程。
当前启动脚本保留源研究环境中已验证的参数，因此操作人员必须在服务器边界完成
这些隔离和 revision 控制。

runner 的命令行参数属于受信任操作员输入，不应直接接收 Web 请求、作业名称或
其他外部字符串；自动化必须把 `--tag` 限定为 `config.MODELS` 中的正式 tag，
并把 `--base-url` 限定为本机或受控 tunnel。共享节点上建议先执行 `umask 077`，
限制 transcript 和 CSV 的默认权限。模型生成的开放题文本属于不受信任数据；
不要直接在会执行公式的电子表格程序中打开 CSV，发布前应按研究数据流程检查。

Google service-account JSON 必须保存在本仓库之外，并限制为当前用户可读。
Sheets 的 Editor 权限作用于整个 workbook，而不只是 `VLM_Responses` tab；对人类
研究数据敏感时，建议先写入独立 workbook 再受控合并。`--replace` 会清空并重写
目标 tab，执行前必须人工核对 spreadsheet 与 tab。

## node03：使用已验证环境

服务器现有 `vlmstudy` 环境的 vLLM 0.19.0 已跑通 Qwen3-VL-8B 完整 pilot，
不要为了其他模型预先重建环境。每个新模型先 pilot；只有真实报出不支持架构时，
再为该模型单独升级或创建环境。

```bash
cd /path/to/v-Means/vlm_userstudy
conda activate vlmstudy
python -m pip install -r requirements.txt
python -c "import vllm, openai; print('vLLM', vllm.__version__)"
python -m unittest discover -s tests -v
```

四个视频放在仓库的 `videos/` 下，文件名必须正好是：

```text
videos/v1_blobs.mp4
videos/v2_cross.mp4
videos/v3_aggregation.mp4
videos/v4_hospital.mp4
```

集成仓库不重复跟踪这四个较大的 MP4；首次运行前执行
`bash download_videos.sh` 下载。`videos/` 中除 `.gitkeep` 外的下载文件和残片
都会被忽略，不会误提交。
正式结果写入
`outputs/vlm_responses.csv`，逐轮审计记录写入 `outputs/raw/`；pilot 只写
`outputs/pilot/`。`outputs/` 已被 Git 忽略，不会误推到 GitHub。

注意：当前 schema 为 Q6 全部 13 项，CSV 是 107 列；旧 12 项 schema 的
pilot CSV 是 103 列。先把旧 `outputs/pilot/` 改名留档，再跑新 pilot；runner
会主动拒绝把两种 schema 混写。

## 每个模型的执行循环

```bash
# 服务窗口（tmux）:
bash serve/<model>.sh                    # node03 卡位已写在每个脚本中
# runner 窗口:
curl -f http://localhost:8000/v1/models  # ready 才继续
python runner.py --tag <tag> --pilot     # 每个新模型必跑，输出隔离在 outputs/pilot/
python score.py outputs/pilot/vlm_responses.csv
# 查看 raw transcript、CSV、格式率和 warnings；不预设通过/失败阈值，
# 结合模型的实际回答决定是否进入正式 runs：
REV=$(python3 -c "from huggingface_hub import HfApi; print(HfApi().model_info('<hf_id>').sha)")
python runner.py --tag <tag> --model-revision $REV      # 正式 3 runs
# 某 run 失败修复后续跑: --start-run-id <N>
# 换模型: 杀 vLLM，换 serve 脚本，重复
```

全部跑完：`python score.py` 看总表；结果回传本地后
`push_to_sheet.py` 写入 Sheet（service-account key 不上服务器）。

## 评分口径（已裁定）

- Q3 按 config 的 `q3_expected`（3 / 8 / 6 / I couldn't tell）；Q4 正确答案 False。
- Q6 主结果用 **human12**：全部 13 项施测，评分排除 `Finding empty space`（计 12 项），
  并将 `Early termination` 判 No；与人类比较时也从人类数据中使用同一 12 项子集。
  **design13** 对全部 13 个施测项作敏感性分析（仅 4 个 distractor 判 No，
  `Early termination` 和 `Finding empty space` 均判 Yes）。
- 缺失/非法答案一律 valid-only 分母，格式服从率单独报告
  （videoFmt / overallFmt 列）。

## 可靠性行为

- 每轮请求：瞬时错误（连接/超时/5xx/429）最多尝试 3 次（等待 30/60s），
  确定性 4xx 立即失败且不空等；受限解码语法被服务器拒绝时自动降级并记录。
- transcript 每轮落盘；会话失败不写 CSV 行；dry-run / pilot 与正式数据完全隔离。
- CSV schema 守卫拒绝新旧表头混写；Sheet 推送默认按
  (model_tag, run_id, timestamp) 去重追加，允许手工尾列，`--replace` 才整表重写。
- 元数据列含 served_model、model_revision、vllm_version、gpu_name、
  serve_script、answer_constraint，保证可复现。

## 目录

```
README.md         独立模块说明、实验协议和运行手册
requirements.txt 本地 runner / Sheet / 下载工具依赖（不含服务端 vLLM）
config.py          视频、q3_expected、运行参数、模型注册表
questionnaire.py   问卷原文 + 选项 + prompt + JSON schema + 校验
runner.py          多轮会话 runner（--pilot / --dry-run / --start-run-id / --model-revision）
score.py           双口径评分 + 格式率
push_to_sheet.py   写 VLM_Responses tab（本地跑）
serve/*.sh         五个正式模型 + 两个明确标为 legacy 的 vLLM 启动脚本
tests/             回归测试（python -m unittest discover -s tests -v）
download_videos.sh yt-dlp 下载四个视频到 videos/（文件名与 config 一致）
videos/.gitkeep    保留素材目录；MP4 由 download_videos.sh 单独获取
```
