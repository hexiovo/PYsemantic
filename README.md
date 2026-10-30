# 中文语义分析工具包

Windows 桌面应用与 Python API，用于在本机导入中文文本、计算向量/词项矩阵、主题、近邻、聚类和文档指标。默认 BGE-small-zh-v1.5 ONNX 权重与分词器已包含在程序中；应用运行时不下载模型、不调用外部 API。

## Windows 便携版

推荐下载 [PYsemantic Windows x64 v1.1.0](https://github.com/hexiovo/PYsemantic/releases/download/v1.1.0/PYsemantic-Windows-x64-v1.1.0.zip)。解压后双击 `PYsemantic.exe` 即可运行，不需要安装 Python 或单独下载模型。此压缩包包含 CPU 与 DirectML 运行时，不包含 CUDA 运行时。首次启动会稍慢，因为单文件程序需要将运行资源解压到 Windows 临时目录。

压缩包内含中文操作指南 PDF、快速开始说明、合成示例 CSV、计算方法说明、来源文献和第三方许可。程序分析不上传文本，也不需要联网；只有在任务页主动安装源码环境依赖，或点击 GPU 资料链接时才会访问网络。推荐先导入压缩包中的 `examples/synthetic_texts.csv` 熟悉流程，再选择自己的文件。

## 源码运行与打包

从 GitHub 获取源码后，先建立 Python 3.12 环境、安装固定依赖并准备模型资源：

```powershell
git clone https://github.com/hexiovo/PYsemantic.git
cd PYsemantic
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe scripts\download_model.py
```

Git 仓库包含源码、分词器和固定模型清单；ONNX 权重、虚拟环境、测试输出和打包 EXE 不进入 Git。模型下载只在源码运行／打包准备时需要联网，完成后应用分析仍完全离线。

开发运行：

```powershell
.\.venv\Scripts\python.exe main.py
```

在 `dist` 目录生成单文件 EXE：

```powershell
pwsh -ExecutionPolicy Bypass -File .\build.ps1
```

也可指定另一个输出目录。完成 PDF 生成和 EXE 打包后，创建可分发的 ZIP：

```powershell
pwsh -ExecutionPolicy Bypass -File .\build.ps1 -OutputDirectory .\dist
pwsh -ExecutionPolicy Bypass -File .\scripts\package_portable.ps1 -Version 1.1.0
```

首次双击时会先显示“PYsemantic · 正在开启…”启动画面，待本地依赖和主窗口准备完成后关闭；后台 worker 不会重复弹出启动画面。

打包前需要完成固定 revision 模型下载（此脚本只供打包准备使用，应用运行时不会调用）：

```powershell
.\.venv\Scripts\python.exe scripts\download_model.py
```

模型 revision、下载 URL、文件字节数和 SHA256 在 `assets/bge-small-zh-v1.5/model-manifest.json`。第三方与模型许可见 `THIRD_PARTY_NOTICES.txt`。

### 重建操作指南 PDF

操作指南的源文件位于 `docs\build_user_guide.py`，所用界面截图保存在 `docs\screenshots`。截图来自真实 Windows 界面；结果页使用应用自带的合成数据烟测。要重建截图，请先运行 `main.py --ui-smoke`，再将对应截图复制到 `docs\screenshots`。在安装了 Noto Sans SC 的 Windows 环境中，可执行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-guide.txt
.\.venv\Scripts\python.exe docs\build_user_guide.py --font 'C:\Windows\Fonts\NotoSansSC-VF.ttf'
```

`scripts\package_portable.ps1` 会将 PDF、可执行文件、快速开始、合成示例、方法说明、来源文献、许可文件与 SHA256 清单装入 Windows x64 ZIP。

### 清理旧的源码工作区输出

源码目录中的 `cleanup-legacy.cmd` 仅用于维护旧构建机。运行后会先列出将清理的 `build` 结果、临时调试文件、旧版 EXE、旧计划文档和 Python 缓存；只有输入 `Y` 才会执行。它会保留当前 `dist\PYsemantic.exe`、发布 ZIP、PDF、校验清单、虚拟环境、模型和测试。请先关闭 PYsemantic；此操作不可撤销。便携 ZIP 不包含此维护脚本。

## 环境检测、依赖安装与计算加速

“任务”页的“检查环境”在后台检查运行依赖的导入状态与版本、本地 BGE 模型文件、当前 Python、CPU 核心数和 ONNX Runtime 后端。缺包时可点击“一键安装缺失依赖”，查看安装输出或取消安装；安装结束会重新检测，请重启程序使新包生效。安装需要联网，只安装本项目清单中的运行依赖，不会下载分析模型。

源码运行时，安装器调用当前 Python 的 pip，使用 `requirements-lock.txt` 中的固定版本。缺少主界面必需包时，程序会先打开仅依赖 Python 标准库的修复窗口。也可手动使用：

```powershell
.\.venv\Scripts\python.exe main.py --check-environment
.\.venv\Scripts\python.exe main.py --repair-environment
.\.venv\Scripts\python.exe main.py --install-missing
```

单文件 EXE 使用内置 Python，安装按钮会禁用，并显示重新安装完整程序包的说明。外部 Python 的 pip 无法修复 EXE 内置依赖；模型文件缺失同样需要按上面的模型准备步骤处理或更换完整程序包。

“方法”页的计算加速设置会随上次设置、命名预设、项目与设置 JSON 保存：

| 设置 | 默认值 | 用途 |
|---|---|---|
| `parallel_enabled` | `false` | 开启单次分析内部的 CPU 并行；关闭时使用数值库默认线程策略 |
| `cpu_workers` | `4` | 开启并行时使用 1–32 个线程；控制数值库、LDA 和 BGE CPU 计算 |
| `bge_device` | `cpu` | 可选 `cpu`、`auto`、`cuda`、`directml`，仅影响 BGE 推理 |

LDA 使用线程并行并限制内部数学库线程，避免叠加占用；多方法任务仍按顺序执行。实际收益取决于数据量与硬件，小数据可能因调度开销变慢，浮点结果也可能有微小差异。

BGE 自动模式按 CUDA、DirectML、CPU 顺序尝试，回退原因和实际活动后端会写入结果提示与运行日志。显式选择 CUDA 或 DirectML 后，如果该后端不可用或会话实际回退到 CPU，任务会报告错误。活动后端表示优先使用该设备，部分算子仍可能由 CPU 执行。

GPU 需要兼容的设备、驱动与 ONNX Runtime 发行包。参阅 [CUDA 安装要求](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) 和 [DirectML 安装要求](https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html)。CPU、CUDA、DirectML 发行包共用 `onnxruntime` 导入名，一个 Python 环境中只保留一种发行包，配置完成后重启程序。应用补包按钮会保留已能导入的 GPU 运行时，不负责切换这些互斥发行包。

### Windows DirectML GPU 打包

在已经安装基础依赖的项目虚拟环境中，关闭正在运行的 Python 分析任务，然后切换到固定的 DirectML 运行库并打包：

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y onnxruntime onnxruntime-gpu
.\.venv\Scripts\python.exe -m pip install -r requirements-directml.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts\gpu_runtime_smoke.py --output build\gpu-validation\source
pwsh -ExecutionPolicy Bypass -File .\build.ps1
```

打包脚本收集 ONNX Runtime 的 DLL 和数据资源，包含 DirectML.dll。在“方法”页将 BGE 设备设为“DirectML”或“自动使用可用 GPU”；CPU 选项仍可使用。同一 EXE 会自动回退到 CPU（自动模式），显式 DirectML 模式则要求对应设备可用。

2026-10-29 在本机 RTX 3050 6GB／驱动 560.94 上，DirectML 1.24.4 的真实 BGE 推理检查通过：125 个 GPU 节点事件、2 个 CPU 节点事件；单条合成文本的 GPU/CPU CLS 向量余弦相似度约 0.9999999999994。此项验证确认设备执行和结果一致性，不代表对实际语料的加速比测量。检查脚本会输出原始 ONNX Runtime profiling 记录与 JSON 摘要。

## 输入

- 粘贴文本、多 TXT 文件或 TXT 目录；TXT 可按每文件、每行或空行分段。
- CSV、TSV、XLSX 和 JSONL；可选文本列与 ID 列。文本编码支持 UTF-8、GB18030、Big5、UTF-16（word2vec 文本向量另支持 UTF-8 / GB18030）。
- 可选空白归一后全文去重；空文本始终排除。重复 ID 保留但增加 `#2`、`#3` 后缀。
- jieba 精确/搜索模式或字符切分；无停词、保守/标准内置自编停词，或每行一词的自定义停词文件；支持 jieba 自定义词典。

## 方法与导出

内置方法：BGE-small-zh-v1.5 ONNX、词 TF-IDF、字符 TF-IDF、LSA、NMF、计数输入 LDA、窗口共现/PPMI、外部文本 word2vec 平均向量。结果包括文档词项计数与 TF-IDF、所选文档表示、可用的完整相似度矩阵、分块近邻、主题矩阵与主题词、词频、KMeans 聚类/适用时的轮廓系数，以及文档字数、词数、TTR。

选择一个父目录后，导出会创建新的唯一子目录，包含 UTF-8 BOM CSV、NPZ 矩阵与标签、离线 HTML 报告、参数 JSON、来源/清洗记录和文献/方法说明。可勾选 XLSX；大型矩阵仍完整保存在 NPZ。外部 word2vec 以流式方式扫描，只保留当前词表命中的向量。

界面支持对同一语料依次运行多种方法、按方法切换结果、导出单项或全部并生成比较表。任务页提供可筛选和导出的完整运行日志、任务历史、已保存项目打开/恢复、命名参数预设和上次设置。项目使用 JSON + NPZ；结果缓存不保留原文，只有用户明确保存的项目会保存语料文本。

方法页可导入或导出设置 JSON；任务页可导入和导出命名预设。设置文件使用版本化 JSON 格式，也兼容旧版上次设置、预设文件和原始 `AnalysisConfig` 字典。导入前会检查字段、类型、选项和取值范围；不可用的本机路径会提示重新选择，不会自动读取输入文本。导入的预设可选择名称后保存到本机预设列表。

## 结果浏览与画像

- **原文联动**：在“文档指标”“近邻”或画像成员列表中，双击一行、按 Enter 或点击“查看原文”。近邻按两栏显示全文、文档 ID、来源和行／段位置。
- **筛选与分页**：文档指标可按 ID 子串和簇筛选；近邻 ID 匹配任一端，簇条件作用于源文档，可设置余弦相似度上下限（-1 至 1，留空不限）。点击“筛选”或在输入框按 Enter 应用，“重置”清除条件。词频可按词语筛选。表格支持每页 50／100／300 条和前后翻页；内容较多时使用表格及页面滚动条。
- **聚类画像**：进入“主题与聚类”，选择“聚类画像”，点击一组查看数量、关键词、代表文本及全部成员。关键词与代表文本数量可设置（默认 10 词、3 篇）；关键词按簇内平均词级 TF-IDF 排序，代表文本按原始表示空间中距簇中心的欧氏距离升序排列。未分簇文档可在“文档指标”筛选查看。
- **自定义簇名**：在画像详情中输入名称并点击“应用名称”，留空恢复默认名称。名称按方法分别保存；修改后到任务页保存项目，下次打开即可恢复。重新运行该方法会重新生成簇，不沿用旧簇名。
- **主题画像**：NMF／LDA 按每篇文档的最大主题权重归属成员（并列取编号较小者），代表文本按该主题权重降序。LSA 分正、负载荷方向展示，各维度成员可能重叠；载荷不是概率。原来的主题词及权重仍可切换到“主题词表”查看。
- **导出**：提供 `group_profiles.csv`（画像与自定义名称）、`group_members.csv`（全部成员及排序指标）和 `group_profiles.json`（名称与解释口径），不额外导出原文。

新分析的原文仅在内存中关联；已保存项目可恢复全文。历史缓存没有原文，会明确显示“原文不可用”，不会按相同 ID 从其他语料猜配原文。

## Python API

```python
from semantic_toolkit.data_io import load_corpus
from semantic_toolkit.analysis import AnalysisConfig, run_analysis
from semantic_toolkit.exports import export_results

corpus = load_corpus(
    [r"D:\data\comments.csv"],
    text_column="评论内容",
    id_column="评论ID",
    deduplicate=True,
)
result = run_analysis(corpus, AnalysisConfig(method="bge"))
print(result.document_metrics.head())
print(result.nearest_neighbors.head())
folder = export_results(result, corpus, r"D:\results", include_xlsx=True)
print(folder)
```

`AnalysisConfig.method` 可选键为 `bge`、`tfidf-word`、`tfidf-char`、`lsa`、`nmf`、`lda`、`cooc-ppmi`、`word2vec`。word2vec 还需传 `word2vec_path`；自定义停词与 jieba 词典分别传 `custom_stopwords_path`、`user_dictionary_path`。

“方法”页下方的“词项与计算参数”提供以下设置，默认值按下表设置：

| 参数 | 默认值 | 范围与作用 |
|---|---|---|
| `max_df` 最大文档比例 | 1.0 | (0, 1]，排除出现文档比例超过此值的词项；与最小文档频次共同作用 |
| `word_ngram_min` / `word_ngram_max` | 1 / 1 | 1–3，连续分词的组合长度，用于词 TF-IDF、LSA、NMF、LDA |
| `char_ngram_min` / `char_ngram_max` | 1 / 2 | 1–5，原文字符组合长度，用于字符 TF-IDF |
| `neighbors_k` 近邻数量 | 10 | 1–100，实际返回数不超过其他非零文档数 |
| `compute_clusters` 计算聚类 | True | 可在方法页关闭 KMeans 与轮廓系数，主题与近邻仍计算 |
| `nmf_max_iter` / `lda_max_iter` | 300 / 20 | 1–1000，分别控制 NMF 与 LDA 最大迭代次数 |
| `random_state` 随机种子 | 42 | 0–4294967295，用于 LSA、NMF、LDA、KMeans 和轮廓系数抽样 |

n-gram 最小值不得大于最大值。组合词在矩阵中以空格连接；使用词组时，词频表统计的是所保留词项（含词组）的出现次数，文档词数和 TTR 按清理后的分词序列统计，不计 n-gram 组合数。BGE、PPMI、word2vec 的辅助词项矩阵保持单词粒度，避免改变词向量匹配和窗口语义。BGE 与字符 TF-IDF 使用保留句子结构的分析文本：分词、停词、最小词长与纯数字 token 过滤仅影响辅助词项统计；用户启用的网址移除或统一小写会改变其输入文本。所有设置随预设、上次设置、项目和 `parameters.json` 保存；旧项目缺失新设置时使用默认值。

方法页还按用途分组提供以下设置：

| 参数 | 默认值 | 范围与作用 |
|---|---|---|
| `min_token_length` | 1 | 1–20 字符，过滤短 token；字符切分时大于 1 会过滤所有单字 token |
| `remove_numbers` / `remove_urls` | False / False | 分别过滤纯数字 token、移除分析文本中的 http(s) / www 网址，含汉字或字母的编号保留 |
| `case_mode` | 方法默认 | 方法默认／统一小写／保持原样；方法默认沿用字符 TF-IDF 小写、其他方法保持原样的行为 |
| `topic_top_words` / `profile_top_words` | 12 / 10 | 各为 1–100，分别控制主题词表和主题／聚类画像的关键词数 |
| `representative_count` | 3 | 1–50，控制画像详情的代表文本数和导出的代表文档 ID 数 |
| `compute_neighbors` / `compute_silhouette` | True / True | 独立开关，关闭后跳过实际计算与 ETA 阶段；轮廓系数还要求开启聚类 |
| `tfidf_sublinear_tf` / `tfidf_smooth_idf` | False / True | 对数词频与 IDF 平滑，用于词／字符 TF-IDF，及相应的外部词向量加权；LDA 输入仍为原始词项计数 |
| `nmf_tol` / `kmeans_tol` | 0.0001 / 0.0001 | 有限正数，分别控制 NMF 和 KMeans 的收敛阈值 |
| `lda_mean_change_tol` | 0.001 | 有限正数，LDA 内部文档主题更新阈值 |
| `lda_evaluate_every` / `lda_perp_tol` | -1 / 0.1 | 评估间隔 -1 关闭，1–100 开启；只有开启时困惑度阈值才参与停止判断，开启评估会增加耗时 |
| `lda_doc_topic_prior` / `lda_topic_word_prior` | None / None | 文档—主题 α 和主题—词 η；填有限正数，或留空使用实际主题数的倒数 |

原始文档及来源不被清理设置覆盖，原文查看仍显示原始内容；“字数”按原始文本统计，“词数”“不同词数”和 TTR 按分析 token 统计。网址识别支持常见 ASCII URL，不是完整的浏览器 URL 解析器。分析清理不改变导入时的全文去重规则。

## 集成检查与界面截图

可从打包好的 EXE 运行以下检查（程序名为 `PYsemantic.exe`）：

```powershell
& '.\dist\PYsemantic.exe' --self-test 'D:\temp\semantic-self-test'
& '.\dist\PYsemantic.exe' --runtime-self-test 'D:\temp\semantic-runtime-self-test'
& '.\dist\PYsemantic.exe' --ui-smoke 'D:\temp\semantic-ui-evidence'
& '.\dist\PYsemantic.exe' --ui-log-smoke 'D:\temp\semantic-log-viewer-evidence'
```

`--self-test` 是分析集成烟测。`--runtime-self-test` 使用小型合成语料检查 worker 阶段、两种方法运行、ETA 初始估计与后续校准、日志和历史、预设、项目保存与往返、取消清理及导出提交；结果写入指定目录，运行配置隔离在该目录的 `test-profile`。

`--ui-smoke` 由应用自身截取五个页面，并用合成 TF-IDF 两方法运行覆盖运行中 ETA、任务管理、历史恢复和项目结果载入；截图和 smoke 专用配置均写入指定目录。它不会运行 BGE 或接管用户鼠标。

`--ui-log-smoke` 单独显示并截取运行日志弹窗，写入合成日志截图和 DPI/窗口尺寸记录；配置与日志只写入指定目录下的 `test-profile`，适合不重跑完整界面烟测时核验日志窗口。

以下命令可在新的输出目录中检查界面、结果和后台任务：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\workbench_ui_smoke.py build\workbench-ui-check
.\.venv\Scripts\python.exe scripts\result_ui_smoke.py build\result-ui-check
.\.venv\Scripts\python.exe scripts\parameter_ui_smoke.py build\parameter-ui-check-new
.\.venv\Scripts\python.exe scripts\advanced_worker_smoke.py dist\PYsemantic.exe build\advanced-worker-check-new
```

结果界面烟测使用 361 篇合成文本，覆盖第八页、筛选、原文对照、画像成员、名称保存恢复、方法切换、无原文缓存及重新导入后的清空行为，并输出截图和 `checks.json`。

ETA API 可通过 `semantic_toolkit.eta.EtaTracker(plan, clock=callable, history=rows)` 注入单调时钟；`semantic_toolkit.runtime.RuntimeStore(root)` 可指定隔离的运行时存储目录。项目 API 为 `save_project` / `load_project_full`，后台 JSONL worker 入口为 `main.py --runtime-worker TASK_DIR`，其中任务请求与结构化事件分别位于 `request.json` 和 `events.jsonl`。

## 资源边界与解释

- 最多 20,000 篇文档、累计 2,000,000 个分析 token、单篇 100,000 个 token；词表最多 20,000 项。
- 完整相似度矩阵最多 1,500 篇文档（float32，约 9 MB）；更大语料仍按 Top-K 设置分块计算近邻（默认 10）。
- 共现词表独立限制为最多 1,000 项；轮廓系数最多抽样 2,000 个非零表示，随机种子可设（默认 42）。
- BGE 对分析文本（默认原文，可选网址／大小写清理）取 CLS 并 L2 归一化；超长文按 510 个正文 token 分块，块向量归一后按 token 数加权平均再归一化，是长文本处理近似。
- 零向量文档会标记，并排除于近邻、聚类和相似度比较；不计作完美自相似。
- 余弦相似度是给定向量空间的几何相似值，不是概率或准确率。内置停词为自编表，并非从语料训练得到。
- PyInstaller onefile 首次启动会把程序资源解压到系统临时目录；Windows 高 DPI 已请求 per-monitor awareness，仍需在目标设备上确认缩放与字体显示。

