from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, label
from recover_ghost_font import best_downward_shift, extract_frames, otsu_threshold, reconstruct

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate a locally moving Ghost Font defense and test the global-shift attack.')
    parser.add_argument('--text', default='GHOST', help='Message rendered in the demo')
    parser.add_argument('--output', type=Path, default=Path('ghost-hardened.mp4'))
    parser.add_argument('--report', type=Path, default=Path('ghost-defense.json'))
    parser.add_argument('--artifacts', type=Path, default=Path('ghost-defense-artifacts'))
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--frames', type=int, default=36)
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--dots', type=int, default=55000)
    parser.add_argument('--tile-size', type=int, default=120)
    parser.add_argument('--amplitude', type=float, default=26.0)
    parser.add_argument('--background-amplitude', type=float, default=36.0, help='Background amplitude (default: 36 px)')
    parser.add_argument('--text-motion', choices=('circular', 'horizontal'), default='horizontal', help='Shared text carrier used for the human motion cue')
    parser.add_argument('--background-motion', choices=('circular', 'vertical'), default='circular', help='Keyed tile-local background carrier')
    parser.add_argument('--seed', type=int, default=1195921235)
    return parser.parse_args()

def discover_bold_font() -> Path:
    candidates = (Path('/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf'), Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'), Path('/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf'), Path('C:/Windows/Fonts/arialbd.ttf'), Path('/System/Library/Fonts/Supplemental/Arial Bold.ttf'))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError('No bold sans-serif font was found.')

def text_mask(text: str, width: int, height: int) -> np.ndarray:
    font_path = discover_bold_font()
    target_width = int(width * 0.72)
    size = max(24, int(height * 0.32))
    while size > 24:
        font = ImageFont.truetype(str(font_path), size)
        probe = ImageDraw.Draw(Image.new('L', (1, 1)))
        box = probe.textbbox((0, 0), text, font=font, stroke_width=2)
        if box[2] - box[0] <= target_width:
            break
        size -= 2
    canvas = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    box = draw.textbbox((0, 0), text, font=font, stroke_width=2)
    x = (width - (box[2] - box[0])) // 2 - box[0]
    y = (height - (box[3] - box[1])) // 2 - box[1]
    draw.text((x, y), text, font=font, fill=255, stroke_width=2, stroke_fill=255)
    return np.asarray(canvas) > 127

def keyed_phase_fields(rng: np.random.Generator, tile_rows: int, tile_cols: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = (tile_rows, tile_cols)
    phase = rng.uniform(0.0, 2.0 * np.pi, shape)
    unit_speed = np.ones(shape, dtype=np.float64)
    return (phase, phase.copy(), unit_speed, unit_speed.copy())

def generate_frames(destination: Path, text: str, width: int, height: int, count: int, fps: int, dot_count: int, tile_size: int, amplitude: float, background_amplitude: float, seed: int, text_motion: str, background_motion: str) -> tuple[list[Path], np.ndarray]:
    destination.mkdir(parents=True, exist_ok=True)
    mask = text_mask(text, width, height)
    rng = np.random.default_rng(seed)
    tile_cols = (width + tile_size - 1) // tile_size
    tile_rows = (height + tile_size - 1) // tile_size
    phase_x, phase_y, speed_x, speed_y = keyed_phase_fields(rng, tile_rows, tile_cols)

    def dot_texture() -> np.ndarray:
        texture = np.full((height, width), 238, dtype=np.uint8)
        x = rng.integers(1, width - 1, size=dot_count, dtype=np.int32)
        y = rng.integers(1, height - 1, size=dot_count, dtype=np.int32)
        texture[y, x] = 35
        texture[y - 1, x] = 70
        texture[y + 1, x] = 70
        texture[y, x - 1] = 70
        texture[y, x + 1] = 70
        return texture
    background_texture = dot_texture()
    foreground_texture = dot_texture()
    frame_paths: list[Path] = []
    for frame_index in range(count):
        time = 2.0 * np.pi * frame_index / max(count, 1)
        if background_motion == 'vertical':
            dx_field = np.zeros_like(phase_x)
            dy_field = background_amplitude * np.sin(speed_y * time + phase_y)
        else:
            dx_field = background_amplitude * np.sin(speed_x * time + phase_x)
            dy_field = background_amplitude * np.cos(speed_y * time + phase_y)
        text_dx = int(round(amplitude * np.sin(time)))
        text_dy = 0 if text_motion == 'horizontal' else int(round(amplitude * np.cos(time)))
        foreground_frame = np.roll(foreground_texture, shift=(text_dy, text_dx), axis=(0, 1))
        frame = np.full((height, width), 238, dtype=np.uint8)
        for tile_y in range(tile_rows):
            y0 = tile_y * tile_size
            y1 = min(y0 + tile_size, height)
            for tile_x in range(tile_cols):
                x0 = tile_x * tile_size
                x1 = min(x0 + tile_size, width)
                dx = int(round(dx_field[tile_y, tile_x]))
                dy = int(round(dy_field[tile_y, tile_x]))
                background = np.roll(background_texture[y0:y1, x0:x1], shift=(dy, dx), axis=(0, 1))
                foreground = foreground_frame[y0:y1, x0:x1]
                local_mask = mask[y0:y1, x0:x1]
                frame[y0:y1, x0:x1] = np.where(local_mask, foreground, background)
        path = destination / f'frame_{frame_index + 1:06d}.png'
        Image.fromarray(frame, 'L').save(path, optimize=True)
        frame_paths.append(path)
    return (frame_paths, mask)

def encode_video(frames: Path, output: Path, fps: int) -> None:
    if shutil.which('ffmpeg') is None:
        raise RuntimeError('ffmpeg was not found on PATH.')
    output.parent.mkdir(parents=True, exist_ok=True)
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-framerate', str(fps), '-i', str(frames / 'frame_%06d.png'), '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', str(output)]
    subprocess.run(command, check=True)

def diagnostic_motion_map(frame_paths: list[Path], start: int=1, end: int=9, max_shift: int=8) -> tuple[np.ndarray, np.ndarray, list[int], float]:
    frames = [np.asarray(Image.open(path).convert('L'), dtype=np.float32) / 255.0 for path in frame_paths[start - 1:end]]
    height, width = frames[0].shape
    residual_sum = np.zeros((height, width), dtype=np.float64)
    residual_count = np.zeros((height, width), dtype=np.uint16)
    shifts: list[int] = []
    first_pair_residual = np.zeros((height, width), dtype=np.float64)
    for pair_index, (first, second) in enumerate(zip(frames, frames[1:])):
        shift = best_downward_shift(first, second, max_shift)
        shifts.append(shift)
        usable = height - shift
        residual = np.abs(first[:usable] - second[shift:])
        residual_sum[:usable] += residual
        residual_count[:usable] += 1
        if pair_index == 0:
            first_pair_residual[:usable] = residual
    mean_residual = residual_sum / np.maximum(residual_count, 1)
    motion_map = gaussian_filter(mean_residual, sigma=4.0)
    margin = max(20, max_shift + 5)
    threshold = otsu_threshold(motion_map[margin:-margin, margin:-margin])
    return (motion_map, first_pair_residual, shifts, threshold)

def save_grayscale_map(values: np.ndarray, path: Path) -> None:
    low, high = np.percentile(values, (1.0, 99.5))
    scaled = np.clip((values - low) / max(high - low, 1e-09), 0.0, 1.0)
    Image.fromarray(np.uint8(scaled * 255), 'L').save(path, optimize=True)

def save_red_mask(mask: np.ndarray, path: Path) -> None:
    image = np.full((*mask.shape, 3), 255, dtype=np.uint8)
    image[mask] = (230, 0, 0)
    Image.fromarray(image, 'RGB').save(path, optimize=True)

def test_global_attack(frame_paths: list[Path], truth: np.ndarray, expected_text: str, artifacts: Path) -> dict[str, object]:
    artifacts.mkdir(parents=True, exist_ok=True)
    motion_map, pair_residual, shifts, threshold = diagnostic_motion_map(frame_paths)
    attack_mask = reconstruct(frame_paths, start=1, end=9, max_shift=8, blur=4.0, min_area=500)
    truth_path = artifacts / 'known_text_mask.png'
    attack_path = artifacts / 'attack_output_full.png'
    save_red_mask(truth, truth_path)
    save_red_mask(attack_mask, attack_path)
    Image.open(frame_paths[0]).save(artifacts / 'single_frame.png')
    Image.open(frame_paths[1]).save(artifacts / 'next_frame.png')
    save_grayscale_map(pair_residual, artifacts / 'registered_pair_residual.png')
    save_grayscale_map(motion_map, artifacts / 'motion_map.png')
    np.save(artifacts / 'motion_map.npy', motion_map.astype(np.float32))
    try:
        from classical_ocr import discover_fonts, recognize
        transcription, _ = recognize(attack_path, discover_fonts())
    except RuntimeError:
        transcription = ''
    components, component_count = label(attack_mask)
    areas = np.bincount(components.ravel())
    retained = [int(area) for area in areas[1:] if area >= 500]
    intersection = int(np.count_nonzero(attack_mask & truth))
    union = int(np.count_nonzero(attack_mask | truth))
    attack_pixels = int(np.count_nonzero(attack_mask))
    truth_pixels = int(np.count_nonzero(truth))
    inside_mean = float(motion_map[truth].mean())
    outside_mean = float(motion_map[~truth].mean())
    return {'attack': 'unchanged nine-frame global vertical registration', 'estimated_shifts': shifts, 'otsu_threshold': threshold, 'retained_component_count': len(retained), 'retained_component_areas': retained, 'attack_mask_fraction': attack_pixels / attack_mask.size, 'precision_against_text': intersection / attack_pixels if attack_pixels else 0.0, 'recall_against_text': intersection / truth_pixels if truth_pixels else 0.0, 'intersection_over_union': intersection / union if union else 0.0, 'mean_residual_inside_text': inside_mean, 'mean_residual_outside_text': outside_mean, 'inside_outside_residual_ratio': inside_mean / outside_mean if outside_mean else 0.0, 'raw_component_count': int(component_count), 'template_readout': transcription, 'expected_text': expected_text.upper(), 'template_readout_matches': transcription == expected_text.upper()}

def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix='hardened-ghost-font-') as temp:
        frame_dir = Path(temp)
        _, truth = generate_frames(frame_dir, text=args.text, width=args.width, height=args.height, count=args.frames, fps=args.fps, dot_count=args.dots, tile_size=args.tile_size, amplitude=args.amplitude, background_amplitude=args.background_amplitude or args.amplitude, seed=args.seed, text_motion=args.text_motion, background_motion=args.background_motion)
        encode_video(frame_dir, args.output, args.fps)
        decoded_dir = Path(temp) / 'decoded'
        decoded_dir.mkdir(parents=True, exist_ok=True)
        decoded_paths = extract_frames(args.output, decoded_dir, frame_limit=9)
        report = test_global_attack(decoded_paths, truth, args.text.upper(), args.artifacts)
    report.update({'text': args.text, 'resolution': [args.width, args.height], 'frames': args.frames, 'fps': args.fps, 'dot_count': args.dots, 'tile_size': args.tile_size, 'amplitude': args.amplitude, 'background_amplitude': args.background_amplitude or args.amplitude, 'text_motion': args.text_motion, 'background_motion': args.background_motion, 'seed': args.seed, 'video': str(args.output), 'measured_from': 'decoded encoded video'})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
if __name__ == '__main__':
    main()
