from __future__ import annotations
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing, binary_opening, gaussian_filter, label

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Recover motion-hidden text by cancelling a vertically moving dot field.')
    parser.add_argument('video', type=Path, help='Input video file in any format supported by FFmpeg')
    parser.add_argument('-o', '--output', type=Path, default=Path('recovered.png'))
    parser.add_argument('--start', type=int, default=1, help='First 1-based frame in the reconstruction window (default: 1)')
    parser.add_argument('--end', type=int, default=9, help='Last 1-based frame in the reconstruction window, inclusive (default: 9)')
    parser.add_argument('--max-shift', type=int, default=8, help='Maximum downward displacement searched between frames (default: 8 px)')
    parser.add_argument('--blur', type=float, default=4.0, help='Gaussian blur used to consolidate the temporal mask (default: 4.0)')
    parser.add_argument('--min-area', type=int, default=500, help='Discard connected mask regions smaller than this many pixels (default: 500)')
    parser.add_argument('--padding', type=int, default=40, help='White padding around the recovered text (default: 40 px)')
    parser.add_argument('--full-frame', action='store_true', help='Keep the original frame dimensions instead of cropping around the text')
    parser.add_argument('--classical-ocr', action='store_true', help='Run deterministic template OCR from classical_ocr.py after reconstruction')
    parser.add_argument('--ocr-output', type=Path, help='Optional OCR text output path (default: output image with a .txt suffix)')
    parser.add_argument('--ocr-font', type=Path, action='append', default=[], help='Bold sans-serif template font for classical OCR; repeat for several candidates')
    return parser.parse_args()

def extract_frames(video: Path, destination: Path, frame_limit: int) -> list[Path]:
    if shutil.which('ffmpeg') is None:
        raise RuntimeError('ffmpeg was not found. Install it and add it to PATH.')
    pattern = destination / 'frame_%06d.png'
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(video), '-frames:v', str(frame_limit), '-vf', 'setpts=N', '-fps_mode', 'passthrough', str(pattern)]
    subprocess.run(command, check=True)
    frames = sorted(destination.glob('frame_*.png'))
    if len(frames) < 2:
        raise RuntimeError('The video produced fewer than two frames.')
    return frames

def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert('L'), dtype=np.float32) / 255.0

def best_downward_shift(a: np.ndarray, b: np.ndarray, max_shift: int) -> int:
    height, width = a.shape
    margin_y = max(10, max_shift + 2)
    margin_x = 10
    best_error = float('inf')
    best_shift = 0
    for dy in range(max_shift + 1):
        y_stop = height - margin_y - dy
        if y_stop <= margin_y:
            continue
        aa = a[margin_y:y_stop, margin_x:width - margin_x]
        bb = b[margin_y + dy:y_stop + dy, margin_x:width - margin_x]
        error = float(np.mean(np.abs(aa - bb)))
        if error < best_error:
            best_error = error
            best_shift = dy
    return best_shift

def otsu_threshold(values: np.ndarray) -> float:
    hist, edges = np.histogram(values, bins=512, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    class_weight = np.cumsum(hist)
    class_sum = np.cumsum(hist * centers)
    total_weight = class_weight[-1]
    total_sum = class_sum[-1]
    denominator = class_weight * (total_weight - class_weight)
    between_class_variance = np.zeros_like(denominator)
    valid = denominator > 0
    between_class_variance[valid] = (total_sum * class_weight[valid] - class_sum[valid] * total_weight) ** 2 / denominator[valid]
    return float(centers[int(np.argmax(between_class_variance))])

def reconstruct(frame_paths: list[Path], start: int, end: int, max_shift: int, blur: float, min_area: int) -> np.ndarray:
    if start < 1 or end <= start or end > len(frame_paths):
        raise ValueError(f'Invalid frame window {start}-{end}; video contains {len(frame_paths)} frames.')
    first = load_gray(frame_paths[start - 1])
    height, width = first.shape
    residual_sum = np.zeros((height, width), dtype=np.float64)
    residual_count = np.zeros((height, width), dtype=np.uint16)
    a = first
    for frame_number in range(start + 1, end + 1):
        b = load_gray(frame_paths[frame_number - 1])
        if b.shape != a.shape:
            raise RuntimeError('The video changes resolution between frames.')
        dy = best_downward_shift(a, b, max_shift)
        usable_height = height - dy
        residual = np.abs(a[:usable_height] - b[dy:])
        residual_sum[:usable_height] += residual
        residual_count[:usable_height] += 1
        a = b
    mean_residual = residual_sum / np.maximum(residual_count, 1)
    motion_map = gaussian_filter(mean_residual, sigma=blur)
    margin = max(20, max_shift + 5)
    core = motion_map[margin:height - margin, margin:width - margin]
    threshold = otsu_threshold(core)
    mask = motion_map > threshold
    mask[:margin] = False
    mask[-margin:] = False
    mask[:, :margin] = False
    mask[:, -margin:] = False
    mask = binary_closing(mask, iterations=1)
    mask = binary_opening(mask, iterations=1)
    components, _ = label(mask)
    areas = np.bincount(components.ravel())
    keep = np.flatnonzero(areas >= min_area)
    keep = keep[keep != 0]
    return np.isin(components, keep)

def render(mask: np.ndarray, output: Path, padding: int, full_frame: bool) -> None:
    height, width = mask.shape
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    image[mask] = (230, 0, 0)
    if not full_frame:
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            raise RuntimeError('No foreground was recovered. Try a different frame window.')
        x0 = max(int(xs.min()) - padding, 0)
        x1 = min(int(xs.max()) + padding + 1, width)
        y0 = max(int(ys.min()) - padding, 0)
        y1 = min(int(ys.max()) + padding + 1, height)
        image = image[y0:y1, x0:x1]
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, 'RGB').save(output, optimize=True)

def main() -> None:
    args = parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    with tempfile.TemporaryDirectory(prefix='ghost-font-') as temp:
        frames = extract_frames(args.video, Path(temp), frame_limit=args.end)
        mask = reconstruct(frames, start=args.start, end=args.end, max_shift=args.max_shift, blur=args.blur, min_area=args.min_area)
        render(mask, args.output, args.padding, args.full_frame)
    print(f'Recovered image: {args.output.resolve()}')
    if args.classical_ocr:
        from classical_ocr import discover_fonts, recognize
        text_output = args.ocr_output or args.output.with_suffix('.txt')
        fonts = args.ocr_font or discover_fonts()
        transcription, details = recognize(args.output, fonts)
        text_output.parent.mkdir(parents=True, exist_ok=True)
        text_output.write_text(transcription + '\n', encoding='utf-8')
        minimum_margin = min((float(item['margin']) for item in details))
        print(f"Classical OCR text: {transcription or '[no text recognized]'}")
        print(f'Minimum template margin: {minimum_margin:.3f}')
        print(f'OCR file: {text_output.resolve()}')
if __name__ == '__main__':
    main()
