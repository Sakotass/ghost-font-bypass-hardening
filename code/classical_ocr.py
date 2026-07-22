from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Iterable
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt, label
ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
NORMALIZED_SIZE = 96

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Classical, non-learning template OCR for a recovered red Ghost Font mask.')
    parser.add_argument('image', type=Path, help='Recovered red-on-white PNG')
    parser.add_argument('--font', type=Path, action='append', default=[], help='Template font file; repeat to provide several candidates')
    parser.add_argument('--output', type=Path, help='Optional text output path')
    parser.add_argument('--min-area', type=int, default=250, help='Minimum connected-component area (default: 250 px)')
    parser.add_argument('--debug-dir', type=Path, help='Optional directory for normalized glyph images')
    return parser.parse_args()

def discover_fonts() -> list[Path]:
    candidates = ['/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', '/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf', '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf', 'C:\\Windows\\Fonts\\arialbd.ttf', 'C:\\Windows\\Fonts\\arial.ttf', 'C:\\Windows\\Fonts\\calibrib.ttf', '/System/Library/Fonts/Supplemental/Arial Bold.ttf', '/System/Library/Fonts/Helvetica.ttc']
    found = []
    for candidate in candidates:
        path = Path(os.path.expandvars(candidate))
        if path.is_file() and path not in found:
            found.append(path)
    return found

def red_mask(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert('RGB'), dtype=np.int16)
    return (rgb[..., 0] > 120) & (rgb[..., 0] > rgb[..., 1] + 60) & (rgb[..., 0] > rgb[..., 2] + 60)

def component_boxes(mask: np.ndarray, min_area: int) -> list[tuple[int, int, int, int]]:
    components, count = label(mask)
    boxes = []
    for component_id in range(1, count + 1):
        ys, xs = np.nonzero(components == component_id)
        if len(xs) < min_area:
            continue
        boxes.append((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    boxes.sort(key=lambda box: box[0])
    return boxes

def normalize(mask: np.ndarray, size: int=NORMALIZED_SIZE, padding: int=6) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return np.zeros((size, size), dtype=bool)
    cropped = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    available = size - 2 * padding
    scale = min(available / cropped.shape[1], available / cropped.shape[0])
    target_w = max(1, int(round(cropped.shape[1] * scale)))
    target_h = max(1, int(round(cropped.shape[0] * scale)))
    resized = Image.fromarray(np.uint8(cropped) * 255).resize((target_w, target_h), Image.Resampling.NEAREST)
    canvas = np.zeros((size, size), dtype=bool)
    x0 = (size - target_w) // 2
    y0 = (size - target_h) // 2
    canvas[y0:y0 + target_h, x0:x0 + target_w] = np.asarray(resized) > 127
    return canvas

def render_template(font_path: Path, character: str, stroke_width: int) -> np.ndarray:
    font = ImageFont.truetype(str(font_path), 180)
    canvas = Image.new('L', (260, 260), 0)
    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), character, font=font, stroke_width=stroke_width)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = (canvas.width - width) // 2 - bbox[0]
    y = (canvas.height - height) // 2 - bbox[1]
    draw.text((x, y), character, font=font, fill=255, stroke_width=stroke_width, stroke_fill=255)
    return normalize(np.asarray(canvas) > 127)

def build_templates(fonts: Iterable[Path]) -> list[tuple[str, Path, int, np.ndarray]]:
    templates = []
    for font_path in fonts:
        for stroke_width in (0, 2, 4, 6, 8):
            for character in ALPHABET:
                templates.append((character, font_path, stroke_width, render_template(font_path, character, stroke_width)))
    return templates

def chamfer_score(observed: np.ndarray, template: np.ndarray) -> float:
    if not observed.any() or not template.any():
        return float('inf')
    distance_to_template = distance_transform_edt(~template)
    distance_to_observed = distance_transform_edt(~observed)
    symmetric_distance = float(distance_to_template[observed].mean()) + float(distance_to_observed[template].mean())
    area_penalty = 10.0 * abs(float(observed.mean()) - float(template.mean()))
    return symmetric_distance + area_penalty

def recognize(image: Path, fonts: list[Path], min_area: int=250, debug_dir: Path | None=None) -> tuple[str, list[dict[str, object]]]:
    mask = red_mask(image)
    boxes = component_boxes(mask, min_area=min_area)
    if not boxes:
        raise RuntimeError('No red glyph components were found.')
    if not fonts:
        raise RuntimeError('No template fonts were found; provide one or more --font paths.')
    templates = build_templates(fonts)
    widths = np.asarray([box[2] - box[0] for box in boxes], dtype=np.float32)
    median_width = float(np.median(widths))
    output = []
    details = []
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
    previous_right = None
    for index, box in enumerate(boxes):
        x0, y0, x1, y1 = box
        if previous_right is not None and x0 - previous_right > 0.55 * median_width:
            output.append(' ')
        glyph = normalize(mask[y0:y1, x0:x1])
        ranked = sorted(((chamfer_score(glyph, template), character, font_path, stroke_width) for character, font_path, stroke_width, template in templates), key=lambda item: item[0])
        best = ranked[0]
        second_distinct = next((item for item in ranked[1:] if item[1] != best[1]))
        confidence_margin = float(second_distinct[0] - best[0])
        output.append(best[1])
        details.append({'index': index, 'box': box, 'character': best[1], 'score': float(best[0]), 'margin': confidence_margin, 'font': str(best[2]), 'stroke_width': int(best[3]), 'runner_up': second_distinct[1]})
        if debug_dir:
            Image.fromarray(np.uint8(glyph) * 255).save(debug_dir / f'glyph_{index:02d}_{best[1]}.png')
        previous_right = x1
    return (''.join(output), details)

def main() -> None:
    args = parse_args()
    fonts = args.font or discover_fonts()
    text, details = recognize(args.image, fonts, min_area=args.min_area, debug_dir=args.debug_dir)
    print(text)
    for detail in details:
        print(f"[{detail['index']}] {detail['character']} score={detail['score']:.3f} margin={detail['margin']:.3f} runner-up={detail['runner_up']} font={Path(detail['font']).name} stroke={detail['stroke_width']}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n', encoding='utf-8')
if __name__ == '__main__':
    main()
