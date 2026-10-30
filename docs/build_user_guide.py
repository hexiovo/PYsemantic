"""Build the Chinese Windows user guide PDF from the checked-in screenshots."""
from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
import tempfile

from fontTools.ttLib import TTFont as FontToolsFont
from fontTools.varLib.instancer import instantiateVariableFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SCREENSHOTS = DOCS / "screenshots"
OUTPUT = DOCS / "PYsemantic操作指南.pdf"

NAVY = colors.HexColor("#15283D")
TEAL = colors.HexColor("#087F82")
INK = colors.HexColor("#24364A")
MUTED = colors.HexColor("#596A7A")
LINE = colors.HexColor("#D5DEE8")
PALE = colors.HexColor("#F3F6F9")
MINT = colors.HexColor("#EAF5F4")
LEMON = colors.HexColor("#F5F8E8")
WHITE = colors.white


def find_font(override: str | None) -> Path:
    choices = [Path(override)] if override else []
    choices.extend([
        Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
    ])
    for candidate in choices:
        if candidate.is_file() and candidate.suffix.lower() == ".ttf":
            return candidate
    raise FileNotFoundError(
        "A Chinese TrueType font is required. On Windows, install Noto Sans SC or pass --font with its TTF path."
    )


def make_styles(font_name: str, bold_font_name: str) -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    styles = {
        "cover_title": ParagraphStyle(
            "CoverTitle", parent=base["Title"], fontName=bold_font_name, fontSize=27,
            leading=37, textColor=NAVY, alignment=TA_LEFT, spaceAfter=7,
            wordWrap="CJK",
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSubtitle", parent=base["Normal"], fontName=font_name, fontSize=12,
            leading=19, textColor=MUTED, alignment=TA_LEFT, spaceAfter=8,
            wordWrap="CJK",
        ),
        "h1": ParagraphStyle(
            "GuideH1", parent=base["Heading1"], fontName=bold_font_name, fontSize=18,
            leading=25, textColor=NAVY, spaceBefore=1, spaceAfter=6,
            wordWrap="CJK",
        ),
        "h2": ParagraphStyle(
            "GuideH2", parent=base["Heading2"], fontName=bold_font_name, fontSize=11.5,
            leading=17, textColor=TEAL, spaceBefore=7, spaceAfter=3,
            wordWrap="CJK",
        ),
        "body": ParagraphStyle(
            "GuideBody", parent=base["BodyText"], fontName=font_name,
            fontSize=9.1, leading=15.3, textColor=INK, spaceAfter=5,
            wordWrap="CJK",
        ),
        "small": ParagraphStyle(
            "GuideSmall", parent=base["BodyText"], fontName=font_name,
            fontSize=7.8, leading=12, textColor=MUTED, spaceAfter=3,
            wordWrap="CJK",
        ),
        "bullet": ParagraphStyle(
            "GuideBullet", parent=base["BodyText"], fontName=font_name,
            fontSize=8.9, leading=14.5, textColor=INK, leftIndent=11,
            firstLineIndent=-9, spaceAfter=3, wordWrap="CJK",
        ),
        "caption": ParagraphStyle(
            "GuideCaption", parent=base["BodyText"], fontName=font_name,
            fontSize=7.6, leading=11, textColor=MUTED, alignment=TA_CENTER,
            spaceBefore=4, spaceAfter=5, wordWrap="CJK",
        ),
        "callout_title": ParagraphStyle(
            "CalloutTitle", parent=base["Heading3"], fontName=bold_font_name,
            fontSize=10, leading=14, textColor=NAVY, spaceAfter=3,
            wordWrap="CJK",
        ),
        "table_head": ParagraphStyle(
            "TableHead", parent=base["BodyText"], fontName=bold_font_name,
            fontSize=8.2, leading=12, textColor=NAVY, wordWrap="CJK",
        ),
        "table_body": ParagraphStyle(
            "TableBody", parent=base["BodyText"], fontName=font_name,
            fontSize=8.1, leading=12, textColor=INK, wordWrap="CJK",
        ),
    }
    return styles


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(text), style)


def bullet(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(f'<font color="#087F82">●</font> {escape(text)}', style)


def callout(title: str, body: str, styles: dict[str, ParagraphStyle], fill=PALE) -> Table:
    table = Table(
        [[para(title, styles["callout_title"]), para(body, styles["body"])]],
        colWidths=[36 * mm, 125 * mm],
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), fill),
        ("BOX", (0, 0), (-1, -1), 0.7, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return table


def fit_screenshot(filename: str, width: float, max_height: float) -> Image:
    path = SCREENSHOTS / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required real UI screenshot is missing: {path}")
    image = Image(str(path))
    factor = min(width / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * factor
    image.drawHeight = image.imageHeight * factor
    image.hAlign = "CENTER"
    return image


def image_block(filename: str, caption: str, width: float, styles: dict[str, ParagraphStyle]) -> list:
    return [
        Spacer(1, 4),
        fit_screenshot(filename, width, 306),
        para(caption, styles["caption"]),
    ]


def method_table(styles: dict[str, ParagraphStyle], width: float) -> Table:
    rows = [
        ["方法", "用途与提示"],
        ["BGE-small-zh-v1.5", "本地中文句向量；默认 CPU，可用 DirectML 时也可让 BGE 在 GPU 上运行。"],
        ["词级 / 字符 TF-IDF", "比较词项或字符组合的分布，适合作为可解释的词汇特征。"],
        ["LSA / NMF / LDA", "把词项矩阵分解成较少的潜在维度或主题；结果需要结合原文解释。"],
        ["共现与 PPMI", "汇总文档内一定窗口的词项共现，反映关联强度，不表示因果关系。"],
        ["外部 word2vec", "平均用户提供的文本向量；仅支持指定格式的文本向量文件。"],
    ]
    data = [[para(cell, styles["table_head"] if row == 0 else styles["table_body"])
             for cell in row] for row in rows]
    table = Table(data, colWidths=[45 * mm, width - 45 * mm], repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E1EBF1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE]),
        ("GRID", (0, 0), (-1, -1), 0.45, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def on_page(canvas, doc) -> None:
    canvas.saveState()
    page_width, page_height = A4
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.6)
    canvas.line(doc.leftMargin, 16 * mm, page_width - doc.rightMargin, 16 * mm)
    canvas.setFillColor(MUTED)
    canvas.setFont("GuideSans", 7.5)
    canvas.drawString(doc.leftMargin, 10.5 * mm, "PYsemantic · 中文语义分析")
    canvas.drawRightString(page_width - doc.rightMargin, 10.5 * mm, f"v1.1.0  ·  第 {doc.page} 页")
    canvas.restoreState()


def build(font_path: Path, output: Path = OUTPUT) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pysemantic-guide-fonts-") as font_temp:
        normal_font = Path(font_path)
        bold_font = Path(font_path)
        variable = FontToolsFont(str(font_path))
        if "fvar" in variable and any(axis.axisTag == "wght" for axis in variable["fvar"].axes):
            axes = {axis.axisTag: axis.defaultValue for axis in variable["fvar"].axes}
            weight = next(axis for axis in variable["fvar"].axes if axis.axisTag == "wght")
            regular_axes = {**axes, "wght": min(max(430, weight.minValue), weight.maxValue)}
            bold_axes = {**axes, "wght": min(max(700, weight.minValue), weight.maxValue)}
            normal_font = Path(font_temp) / "guide-regular.ttf"
            bold_font = Path(font_temp) / "guide-bold.ttf"
            instantiateVariableFont(FontToolsFont(str(font_path)), regular_axes, inplace=True).save(normal_font)
            instantiateVariableFont(FontToolsFont(str(font_path)), bold_axes, inplace=True).save(bold_font)
        pdfmetrics.registerFont(TTFont("GuideSans", str(normal_font)))
        pdfmetrics.registerFont(TTFont("GuideSansBold", str(bold_font)))
        pdfmetrics.registerFontFamily("GuideSans", normal="GuideSans", bold="GuideSansBold", italic="GuideSans", boldItalic="GuideSansBold")
        styles = make_styles("GuideSans", "GuideSansBold")
        return build_pdf(output, styles)


def build_pdf(output: Path, styles: dict[str, ParagraphStyle]) -> Path:
    doc = SimpleDocTemplate(
        str(output), pagesize=A4, rightMargin=21 * mm, leftMargin=21 * mm,
        topMargin=18 * mm, bottomMargin=21 * mm,
        title="PYsemantic 中文语义分析工具操作指南",
        author="PYsemantic",
        subject="Windows 便携版使用说明",
    )
    width = doc.width
    story = [
        Spacer(1, 24 * mm),
        para("PYsemantic", styles["cover_title"]),
        para("中文语义分析工具 · Windows 便携版操作指南", styles["cover_subtitle"]),
        Spacer(1, 2 * mm),
        callout(
            "适合第一次使用",
            "完整解压压缩包，双击 PYsemantic.exe；在“01 数据准备”导入文本，在“02 方法设置”选择方法，然后开始分析。程序和内置 BGE 模型无需安装 Python，下载后可离线分析。",
            styles, MINT,
        ),
        Spacer(1, 10 * mm),
        para("建议的操作顺序", styles["h2"]),
        bullet("先用压缩包中的 examples/synthetic_texts.csv 熟悉导入、计算和导出流程。它只包含自动编写的合成示例，不是研究数据。", styles["bullet"]),
        bullet("确认每行、每段或每个文件分别代表什么，再选择文本列、ID 列与方法。", styles["bullet"]),
        bullet("计算完成后检查原文、参数和导出记录；需要恢复时可保存项目。", styles["bullet"]),
        Spacer(1, 8 * mm),
        callout(
            "便携版设备支持",
            "本版包含 CPU 和 DirectML 运行时。CPU 是默认设备；DirectML 需要兼容的 GPU 与驱动。CUDA 运行时不包含在此 ZIP 中。首次启动可能较慢，因为 EXE 会先解压内置模型与运行库。",
            styles, LEMON,
        ),
        Spacer(1, 13 * mm),
        para("本指南按界面 01–05 页组织，截图来自真实 Windows 应用窗口。用于展示结果的文本由程序生成，仅说明操作位置，不代表任何数据分析结论。", styles["small"]),
        PageBreak(),
        para("01  数据准备：导入与清理", styles["h1"]),
        para("先确定分析单位，再导入文本。界面显示读取数量、空值排除与清理摘要，原始输入文件保持只读。", styles["body"]),
        bullet("粘贴文本时选择切分方式：整段作为一篇、每行一篇，或按空行分成多篇；导入 TXT 文件也使用相同切分规则。", styles["bullet"]),
        bullet("文件输入支持 TXT、CSV、TSV、XLSX 和 JSONL；TXT 目录会读取其中的文本文件。表格可指定文本列与 ID 列。", styles["bullet"]),
        bullet("空文本会排除。重复 ID 会加后缀以保持唯一；只有勾选去重时才会按归一空白后的全文删除重复文本。", styles["bullet"]),
        bullet("示例 CSV 的列名为 text 与 doc_id：选择文件后将 text 设为文本列、doc_id 设为 ID 列，再点“读取并清理”。", styles["bullet"]),
        *image_block("01-data.png", "01 数据准备页。可粘贴文字、选择文件或读取 TXT 目录。", width, styles),
        PageBreak(),
        para("02  方法设置：选择表示与设备", styles["h1"]),
        para("方法会决定文本如何转成向量或词项矩阵。可先使用默认值完成一轮分析，再根据问题逐项调整参数。", styles["body"]),
        method_table(styles, width),
        Spacer(1, 5),
        bullet("BGE 的设备选项只影响 BGE ONNX 推理；其它方法仍按各自的 CPU 实现计算。CPU 并行开关单独控制相关数值计算，默认关闭。", styles["bullet"]),
        bullet("选择“自动使用可用 GPU”时会尝试可用的 provider 并回退到 CPU。显式选择 DirectML 要求驱动与设备可用；GPU 加速效果取决于语料规模。", styles["bullet"]),
        *image_block("02-methods.png", "02 方法设置页。截图显示默认 CPU 设置；页面下方还有分词、停词和计算参数。", width, styles),
        PageBreak(),
        para("开始分析与查看结果", styles["h1"]),
        para("选择需要的方法后点击“开始分析”。耗时计算在后台进行，任务页显示当前阶段和进度；任务完成后可在结果页按方法切换输出。", styles["body"]),
        bullet("结果页包含概览、文档指标、词频、近邻、主题与聚类、相似度热图。双击文档或近邻行，或选中后按 Enter，可查看原文。", styles["bullet"]),
        bullet("表格可按 ID、簇或相似度筛选，并按每页 50、100 或 300 条分页。完整导出不受界面预览行数限制。", styles["bullet"]),
        bullet("完整文档相似度矩阵最多支持 1,500 篇文档；更大语料仍可按 Top-K 分块计算近邻。", styles["bullet"]),
        *image_block("03-results.png", "03 结果查看页。截图使用六篇自动生成的合成文本，展示词级与字符级 TF-IDF 结果。", width, styles),
        PageBreak(),
        para("导出结果、保存项目与管理任务", styles["h1"]),
        bullet("“导出选定结果”仅导出当前方法；“导出全部与比较表”导出所有已完成方法。选择父目录后，程序创建新的结果子目录。", styles["bullet"]),
        bullet("结果包按方法包含适用的 UTF-8 CSV、NPZ 矩阵与轴标签、参数 JSON、数据处理记录和离线 HTML 报告；可勾选 XLSX。", styles["bullet"]),
        bullet("点击“保存当前项目”会把语料文本、清理记录、设置和已完成结果写入所选文件夹。只有主动保存项目时，项目包才会包含原文。", styles["bullet"]),
        bullet("任务页可打开已保存项目或历史结果、查看和导出日志、检查运行环境，以及保存/载入命名参数预设。设置 JSON 记录参数与路径，不会导入文本。", styles["bullet"]),
        *image_block("04-tasks.png", "04 任务页上部。页面可向下滚动查看日志、项目、历史记录和参数预设。", width, styles),
        PageBreak(),
        para("05  帮助与方法资料", styles["h1"]),
        para("帮助页顶部提供“打开操作指南 PDF”和“另存操作指南”按钮。打开时使用 Windows 当前的 PDF 阅读器；另存可将手册复制到指定位置，便于归档或分享。", styles["body"]),
        bullet("“计算口径与方法”说明各指标、方法和资源限制。", styles["bullet"]),
        bullet("“来源文献”列出模型、方法与软件依据；点击文献链接会打开系统浏览器。", styles["bullet"]),
        bullet("“第三方许可”显示模型和运行依赖的许可信息。GPU 资料链接只有点击后才会打开浏览器。", styles["bullet"]),
        *image_block("05-help.png", "05 帮助页。操作指南按钮位于其它方法、文献与许可链接之前。", width, styles),
        PageBreak(),
        para("结果解释、隐私与常见问题", styles["h1"]),
        para("结果表示的是选定方法与参数下的文本特征。余弦相似度不是概率、准确率或通用的重复判定阈值；聚类画像用于探索分组，主题和聚类都不是人工标注的真值。", styles["body"]),
        callout(
            "计算范围",
            "单次分析最多 20,000 篇文档、累计 2,000,000 个分析 token、单篇 100,000 个 token。完整相似度矩阵限于 1,500 篇文档；较大语料仍可使用分块近邻。超过范围时按界面提示拆分数据。",
            styles, PALE,
        ),
        Spacer(1, 7),
        para("常见问题", styles["h2"]),
        bullet("首次启动较慢：单文件程序会先解压运行库和模型；之后启动通常更快。", styles["bullet"]),
        bullet("无法分析或检查到缺少模型：请确认已完整解压，并重新下载完整压缩包。EXE 内置环境不能通过任务页的依赖安装按钮修改。", styles["bullet"]),
        bullet("DirectML 不可用：检查设备、驱动及环境页列出的 provider。自动模式会回退 CPU；显式 DirectML 在设备不兼容时会报错。", styles["bullet"]),
        bullet("任务耗时较长：在“04 任务”查看阶段进度；可点击“取消”。部分模型调用需等当前批次完成后才停止。", styles["bullet"]),
        Spacer(1, 5),
        para("隐私与本地文件", styles["h2"]),
        para("PYsemantic 在本机运行分析，不上传文本，也不会在启动时联网。运行日志和结果缓存保存在当前 Windows 用户的数据目录；普通运行日志不记录粘贴文本。主动导出结果或保存项目时才向用户选择的位置写入相应文件。项目包包含原始语料，保存与分享前请检查其中的数据。", styles["body"]),
        para("离线分析使用内置 BGE 模型。程序不会自动下载模型；CUDA 或 DirectML 驱动、第三方 PDF 阅读器与外部浏览器属于本机环境。打开来源链接或 GPU 资料链接时，网络访问由用户点击启动的浏览器完成。", styles["body"]),
        Spacer(1, 4),
        callout(
            "字体说明",
            "本指南 PDF 嵌入了用于显示中文的 Noto Sans SC 字形子集。该字体版权归 Adobe，采用 SIL Open Font License 1.1；字体文件不作为单独文件随软件包分发。",
            styles, MINT,
        ),
    ]
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PYsemantic's Chinese user guide PDF.")
    parser.add_argument("--font", help="Path to a Chinese TrueType font. Defaults to Noto Sans SC on Windows.")
    parser.add_argument("--output", type=Path, default=OUTPUT, help="PDF output path.")
    args = parser.parse_args()
    font_path = find_font(args.font)
    print(build(font_path, args.output.resolve()))


if __name__ == "__main__":
    main()
