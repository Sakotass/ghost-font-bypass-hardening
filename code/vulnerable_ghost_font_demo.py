from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, label
from classical_ocr import discover_fonts, recognize
from hardened_ghost_font_demo import text_mask
from recover_ghost_font import best_downward_shift, extract_frames, otsu_threshold, reconstruct, render

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate a globally moving vulnerable baseline and attack it.')
    parser.add_argument('--text', default='GHOST')
    parser.add_argument('--output', type=Path, default=Path('ghost-vulnerable.mp4'))
    parser.add_argument('--report', type=Path, default=Path('ghost-attack.json'))
    parser.add_argument('--artifacts', type=Path, default=Path('ghost-attack-artifacts'))
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--frames', type=int, default=48)
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--dots', type=int, default=70000)
    parser.add_argument('--background-speed', type=int, default=4)
    parser.add_argument('--text-speed', type=int, default=-2)
    parser.add_argument('--seed', type=int, default=1195921235)
    return parser.parse_args()

def dot_texture(rng: np.random.Generator, width: int, height: int, dot_count: int) -> np.ndarray:
    texture = np.full((height, width), 238, dtype=np.uint8)
    x = rng.integers(1, width - 1, size=dot_count, dtype=np.int32)
    y = rng.integers(1, height - 1, size=dot_count, dtype=np.int32)
    texture[y, x] = 35
    texture[y - 1, x] = 70
    texture[y + 1, x] = 70
    texture[y, x - 1] = 70
    texture[y, x + 1] = 70
    return texture

def generate_frames(destination: Path, text: str, width: int, height: int, count: int, dot_count: int, background_speed: int, text_speed: int, seed: int) -> tuple[list[Path], np.ndarray]:
    destination.mkdir(parents=True, exist_ok=True)
    truth = text_mask(text, width, height)
    rng = np.random.default_rng(seed)
    background_texture = dot_texture(rng, width, height, dot_count)
    foreground_texture = dot_texture(rng, width, height, dot_count)
    paths: list[Path] = []
    for index in range(count):
        background = np.roll(background_texture, background_speed * index, axis=0)
        foreground = np.roll(foreground_texture, text_speed * index, axis=0)
        frame = np.where(truth, foreground, background).astype(np.uint8)
        path = destination / f'frame_{index + 1:06d}.png'
        Image.fromarray(frame, 'L').save(path, optimize=True)
        paths.append(path)
    return (paths, truth)

def encode_video(frame_dir: Path, output: Path, fps: int) -> None:
    if shutil.which('ffmpeg') is None:
        raise RuntimeError('ffmpeg was not found on PATH')
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-framerate', str(fps), '-i', str(frame_dir / 'frame_%06d.png'), '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', str(output)], check=True)

def motion_diagnostics(paths: list[Path], start: int=1, end: int=9, max_shift: int=8) -> tuple[np.ndarray, np.ndarray, list[int], float]:
    frames = [np.asarray(Image.open(path).convert('L'), dtype=np.float32) / 255.0 for path in paths[start - 1:end]]
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
    image[mask] = (224, 0, 0)
    Image.fromarray(image, 'RGB').save(path, optimize=True)

def main() -> None:
    args = parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='vulnerable-ghost-font-') as temp:
        temp_root = Path(temp)
        source_dir = temp_root / 'source'
        _, truth = generate_frames(source_dir, text=args.text, width=args.width, height=args.height, count=args.frames, dot_count=args.dots, background_speed=args.background_speed, text_speed=args.text_speed, seed=args.seed)
        encode_video(source_dir, args.output, args.fps)
        decoded_dir = temp_root / 'decoded'
        decoded_dir.mkdir(parents=True, exist_ok=True)
        decoded_paths = extract_frames(args.output, decoded_dir, 9)
        motion_map, pair_residual, shifts, threshold = motion_diagnostics(decoded_paths)
        attack_mask = reconstruct(decoded_paths, 1, 9, 8, 4.0, 500)
        Image.open(decoded_paths[0]).save(args.artifacts / 'single_frame.png')
        Image.open(decoded_paths[1]).save(args.artifacts / 'next_frame.png')
        save_grayscale_map(pair_residual, args.artifacts / 'registered_pair_residual.png')
        save_grayscale_map(motion_map, args.artifacts / 'motion_map.png')
        np.save(args.artifacts / 'motion_map.npy', motion_map.astype(np.float32))
        save_red_mask(truth, args.artifacts / 'known_text_mask.png')
        save_red_mask(attack_mask, args.artifacts / 'attack_output_full.png')
        render(attack_mask, args.artifacts / 'recovered_red.png', 40, False)
    transcription, details = recognize(args.artifacts / 'recovered_red.png', discover_fonts())
    components, count = label(attack_mask)
    areas = np.bincount(components.ravel())
    retained = [int(area) for area in areas[1:] if area >= 500]
    intersection = int(np.count_nonzero(attack_mask & truth))
    union = int(np.count_nonzero(attack_mask | truth))
    attack_pixels = int(np.count_nonzero(attack_mask))
    inside_mean = float(motion_map[truth].mean())
    outside_mean = float(motion_map[~truth].mean())
    report = {'demo': 'reproducible vulnerable global-motion baseline', 'text': args.text.upper(), 'resolution': [args.width, args.height], 'frames': args.frames, 'fps': args.fps, 'dot_count': args.dots, 'background_speed_px_per_frame': args.background_speed, 'text_speed_px_per_frame': args.text_speed, 'seed': args.seed, 'attack_frames': 9, 'estimated_shifts': shifts, 'otsu_threshold': threshold, 'retained_component_count': len(retained), 'retained_component_areas': retained, 'attack_mask_fraction': attack_pixels / attack_mask.size, 'intersection_over_union': intersection / union if union else 0.0, 'mean_residual_inside_text': inside_mean, 'mean_residual_outside_text': outside_mean, 'inside_outside_residual_ratio': inside_mean / outside_mean if outside_mean else 0.0, 'template_readout': transcription, 'template_readout_matches': transcription == args.text.upper(), 'minimum_template_margin': min((float(item['margin']) for item in details)), 'video': str(args.output)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
if __name__ == '__main__':
    main()
