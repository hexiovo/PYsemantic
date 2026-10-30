"""Generate the concise, branded startup splash bundled by PyInstaller."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


NAVY = "#15283D"
INK = "#182B43"
TEAL = "#087F82"
CANVAS = "#F3F6F9"
WHITE = "#FFFFFF"
LINE = "#D5DEE8"
MUTED = "#5C6B7B"


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        [r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\seguisb.ttf"] if bold else
        [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\segoeui.ttf"]
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def main() -> Path:
    root = Path(__file__).resolve().parents[1]
    output = root / "assets" / "startup-splash.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    width, height = 900, 460
    image = Image.new("RGB", (width, height), CANVAS)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((24, 24, width - 24, height - 24), radius=28, fill=WHITE, outline=LINE, width=2)

    mark = (80, 76, 156, 152)
    draw.rounded_rectangle(mark, radius=18, fill=NAVY)
    # Three text strokes and a linking point mirror the app icon at splash scale.
    draw.rounded_rectangle((97, 93, 123, 98), radius=2, fill=WHITE)
    draw.rounded_rectangle((97, 107, 123, 112), radius=2, fill=WHITE)
    draw.rounded_rectangle((97, 121, 117, 126), radius=2, fill=WHITE)
    draw.line((126, 113, 144, 113), fill=TEAL, width=5)
    draw.ellipse((138, 105, 154, 121), fill=TEAL)

    draw.text((184, 84), "PYsemantic", font=_font(42, bold=True), fill=INK)
    draw.text((187, 137), "中文语义分析工作台", font=_font(20), fill=MUTED)
    draw.line((80, 196, width - 80, 196), fill=LINE, width=2)

    draw.text((80, 234), "正在开启…", font=_font(31, bold=True), fill=INK)
    draw.text((82, 286), "正在加载本地组件，请稍候", font=_font(20), fill=MUTED)

    draw.rounded_rectangle((82, 356, 818, 357), radius=1, fill=LINE)
    draw.text((82, 377), "所有数据均在本机处理", font=_font(15), fill=MUTED)
    image.save(output, format="PNG", optimize=True)
    return output


if __name__ == "__main__":
    print(main())
