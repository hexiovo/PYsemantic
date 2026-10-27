# 中文语义分析工具包

Windows 桌面应用与 Python API，用于在本机导入中文文本、计算向量/词项矩阵、主题、近邻、聚类和文档指标。默认 BGE-small-zh-v1.5 ONNX 权重与分词器已包含在程序中；应用运行时不下载模型、不调用外部 API。

## 运行与打包

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

生成单文件 EXE 并复制到 `C:\Users\DELL\Desktop\APP\ChineseSemanticToolkit.exe`：

```powershell
pwsh -ExecutionPolicy Bypass -File .\build.ps1
```

打包前需要完成固定 revision 模型下载（此脚本只供打包准备使用，应用运行时不会调用）：

```powershell
.\.venv\Scripts\python.exe scripts\download_model.py
```

模型 revision、下载 URL、文件字节数和 SHA256 在 `assets/bge-small-zh-v1.5/model-manifest.json`。第三方与模型许可见 `THIRD_PARTY_NOTICES.txt`。

## 输入

- 粘贴文本、多 TXT 文件或 TXT 目录；TXT 可按每文件、每行或空行分段。
- CSV、TSV、XLSX 和 JSONL；可选文本列与 ID 列。文本编码支持 UTF-8、GB18030、Big5、UTF-16（word2vec 文本向量另支持 UTF-8 / GB18030）。
- 可选空白归一后全文去重；空文本始终排除。重复 ID 保留但增加 `#2`、`#3` 后缀。
- jieba 精确/搜索模式或字符切分；无停词、保守/标准内置自编停词，或每行一词的自定义停词文件；支持 jieba 自定义词典。

## 方法与导出

内置方法：BGE-small-zh-v1.5 ONNX、词 TF-IDF、字符 TF-IDF、LSA、NMF、计数输入 LDA、窗口共现/PPMI、外部文本 word2vec 平均向量。结果包括文档词项计数与 TF-IDF、所选文档表示、可用的完整相似度矩阵、分块近邻、主题矩阵与主题词、词频、KMeans 聚类/适用时的轮廓系数，以及文档字数、词数、TTR。

选择一个父目录后，导出会创建新的唯一子目录，包含 UTF-8 BOM CSV、NPZ 矩阵与标签、离线 HTML 报告、参数 JSON、来源/清洗记录和文献/方法说明。可勾选 XLSX；大型矩阵仍完整保存在 NPZ。外部 word2vec 以流式方式扫描，只保留当前词表命中的向量。

界面支持对同一语料依次运行多种方法、按方法切换结果、导出单项或全部并生成比较表。任务页提供可筛选和导出的完整运行日志、任务历史、已保存项目打开/恢复、命名参数预设和上次设置。项目使用 JSON + NPZ；结果缓存不保留原文，只有用户明确保存的项目会保存语料文本。

## 结果浏览与画像

- **原文联动**：在“文档指标”“近邻”或画像成员列表中，双击一行、按 Enter 或点击“查看原文”。近邻按两栏显示全文、文档 ID、来源和行／段位置。
- **筛选与分页**：文档指标可按 ID 子串和簇筛选；近邻 ID 匹配任一端，簇条件作用于源文档，可设置余弦相似度上下限（-1 至 1，留空不限）。点击“筛选”或在输入框按 Enter 应用，“重置”清除条件。词频可按词语筛选。表格支持每页 50／100／300 条和前后翻页，不再只显示前 300 条；内容较多时使用表格及页面滚动条。
- **聚类画像**：进入“主题与聚类”，选择“聚类画像”，点击一组查看数量、关键词、代表文本及全部成员。关键词与代表文本数量可设置（默认 10 词、3 篇）；关键词按簇内平均词级 TF-IDF 排序，代表文本按原始表示空间中距簇中心的欧氏距离升序排列。未分簇文档可在“文档指标”筛选查看。
- **自定义簇名**：在画像详情中输入名称并点击“应用名称”，留空恢复默认名称。名称按方法分别保存；修改后到任务页保存项目，下次打开即可恢复。重新运行该方法会重新生成簇，不沿用旧簇名。
- **主题画像**：NMF／LDA 按每篇文档的最大主题权重归属成员（并列取编号较小者），代表文本按该主题权重降序。LSA 分正、负载荷方向展示，各维度成员可能重叠；载荷不是概率。原来的主题词及权重仍可切换到“主题词表”查看。
- **导出**：新增 `group_profiles.csv`（画像与自定义名称）、`group_members.csv`（全部成员及排序指标）和 `group_profiles.json`（名称与解释口径），不额外导出原文。

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

“方法”页下方的“词项与计算参数”提供以下设置，默认值沿用旧版计算行为：

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

## 集成烟测与界面截图

可从 EXE 运行：

```powershell
& 'C:\Users\DELL\Desktop\APP\ChineseSemanticToolkit.exe' --self-test 'D:\temp\semantic-self-test'
& 'C:\Users\DELL\Desktop\APP\ChineseSemanticToolkit.exe' --runtime-self-test 'D:\temp\semantic-runtime-self-test'
& 'C:\Users\DELL\Desktop\APP\ChineseSemanticToolkit.exe' --ui-smoke 'D:\temp\semantic-ui-evidence'
& 'C:\Users\DELL\Desktop\APP\ChineseSemanticToolkit.exe' --ui-log-smoke 'D:\temp\semantic-log-viewer-evidence'
```

`--self-test` 是原有分析集成烟测。`--runtime-self-test` 使用小型合成语料检查真实 worker 阶段、两方法运行、ETA 冷启动与观测校准、日志/历史/预设、项目保存与往返、取消清理和导出原子提交；结果写入指定目录，运行配置隔离在该目录的 `test-profile`。

`--ui-smoke` 由应用自身截取五个页面，并用合成 TF-IDF 两方法运行覆盖运行中 ETA、任务管理、历史恢复和项目结果载入；截图和 smoke 专用配置均写入指定目录。它不会运行 BGE 或接管用户鼠标。

`--ui-log-smoke` 单独显示并截取运行日志弹窗，写入合成日志截图和 DPI/窗口尺寸记录；配置与日志只写入指定目录下的 `test-profile`，适合不重跑完整界面烟测时核验日志窗口。

新增功能的验证命令（使用新的输出目录）：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts\result_ui_smoke.py build\result-ui-check
.\.venv\Scripts\python.exe scripts\parameter_ui_smoke.py build\parameter-ui-check-new
.\.venv\Scripts\python.exe scripts\advanced_worker_smoke.py dist\ChineseSemanticToolkit.exe build\advanced-worker-check-new
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

