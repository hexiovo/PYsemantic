"""Generate an original crisp geometric multi-frame Windows icon."""
from pathlib import Path
from PIL import Image, ImageDraw


def main() -> Path:
    root = Path(__file__).resolve().parents[1]
    output = root / "assets" / "app.ico"
    output.parent.mkdir(parents=True, exist_ok=True)
    scale = 4
    s = 256 * scale
    image = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    px = lambda value: int(value * scale)
    draw.rounded_rectangle((px(8), px(8), px(248), px(248)), radius=px(50), fill="#15283D", outline="#15283D", width=px(4))
    draw.rounded_rectangle((px(35), px(35), px(221), px(221)), radius=px(30), fill="#F3F6F9")
    # Three connected document tiles suggest text fragments becoming a vector.
    draw.rounded_rectangle((px(54), px(61), px(121), px(136)), radius=px(12), fill="#FFFFFF", outline="#D5DEE8", width=px(4))
    draw.rounded_rectangle((px(134), px(61), px(201), px(136)), radius=px(12), fill="#FFFFFF", outline="#D5DEE8", width=px(4))
    draw.line((px(72), px(83), px(104), px(83)), fill="#182B43", width=px(5))
    draw.line((px(72), px(100), px(104), px(100)), fill="#182B43", width=px(5))
    draw.line((px(152), px(83), px(184), px(83)), fill="#182B43", width=px(5))
    draw.line((px(152), px(100), px(184), px(100)), fill="#182B43", width=px(5))
    draw.line((px(121), px(102), px(134), px(102)), fill="#182B43", width=px(8))
    draw.ellipse((px(109), px(154), px(147), px(192)), fill="#087F82", outline="#182B43", width=px(5))
    draw.polygon([(px(153), px(174)), (px(187), px(154)), (px(189), px(192))], fill="#087F82", outline="#182B43")
    image = image.resize((256, 256), Image.Resampling.LANCZOS)
    sizes = [(16,16),(20,20),(24,24),(30,30),(32,32),(36,36),(40,40),(44,44),(48,48),(64,64),(96,96),(128,128),(256,256)]
    image.save(output, format="ICO", sizes=sizes)
    return output


if __name__ == "__main__":
    print(main())

