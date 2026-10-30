# 操作指南构建文件

- `build_user_guide.py` 使用 ReportLab 生成 `PYsemantic操作指南.pdf`。
- `screenshots/` 保存 PDF 引用的五张真实 Windows 界面截图。
- 截图来自 `main.py --ui-smoke`：输入、方法、任务和帮助页是当前程序窗口；结果页展示六篇合成文本的 TF-IDF 分析结果。
- 重建截图时，将烟测输出中的 `01-data.png`、`02-methods.png`、`04-tasks.png`、`05-help.png` 和 `08-project-restored-results.png` 分别复制为 `01-data.png`、`02-methods.png`、`04-tasks.png`、`05-help.png` 和 `03-results.png`。
- PDF 构建需要 `reportlab`、`fonttools` 和中文 TrueType 字体。Windows 可使用 Noto Sans SC：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-guide.txt
.\.venv\Scripts\python.exe docs\build_user_guide.py --font 'C:\Windows\Fonts\NotoSansSC-VF.ttf'
```
