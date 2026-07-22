from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from PIL import Image

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build the measured attack figure.')
    parser.add_argument('--results', type=Path, default=Path('attack-demo'))
    parser.add_argument('--output', type=Path, default=Path('figures'))
    return parser.parse_args()

def heatmap(image: np.ndarray) -> np.ndarray:
    values = image.astype(np.float32) / 255.0
    return np.uint8(colormaps['magma'](values)[..., :3] * 255)

def aligned_composite(first: np.ndarray, second: np.ndarray, shift: int) -> np.ndarray:
    aligned = np.full_like(second, 238)
    if shift:
        aligned[:-shift] = second[shift:]
    else:
        aligned[:] = second
    return np.stack((first, aligned, aligned), axis=-1)

def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(0.025, 0.94, label, transform=axis.transAxes, va='top', ha='left', fontsize=7.5, fontweight='bold', color='white', bbox={'boxstyle': 'square,pad=0.15', 'facecolor': 'black', 'edgecolor': 'none'})

def build(results: Path, output: Path) -> None:
    artifacts = results / 'artifacts'
    report = json.loads((results / 'ghost-attack.json').read_text(encoding='utf-8'))
    first = np.asarray(Image.open(artifacts / 'single_frame.png').convert('L'))
    second = np.asarray(Image.open(artifacts / 'next_frame.png').convert('L'))
    pair = np.asarray(Image.open(artifacts / 'registered_pair_residual.png').convert('L'))
    motion = np.asarray(Image.open(artifacts / 'motion_map.png').convert('L'))
    recovered = np.asarray(Image.open(artifacts / 'attack_output_full.png').convert('RGB'))
    shift = int(report['estimated_shifts'][0])
    panels = [(first, 'capture', 'frame 1'), (aligned_composite(first, second, shift), 'register', f'background: {shift} px'), (heatmap(pair), 'subtract', 'one aligned pair'), (heatmap(motion), 'accumulate', 'eight pairs'), (recovered, 'extract', 'threshold + clean')]
    figure, axes = plt.subplots(1, 5, figsize=(9.4, 2.05), constrained_layout=True)
    for index, (axis, (image, title, subtitle)) in enumerate(zip(axes, panels)):
        axis.imshow(image, cmap='gray', vmin=0, vmax=255)
        axis.set_title(title, fontsize=8, fontweight='bold', pad=2)
        axis.set_xlabel(subtitle, fontsize=6.5, labelpad=2)
        axis.set_xticks([])
        axis.set_yticks([])
        panel_label(axis, chr(ord('a') + index))
        for spine in axis.spines.values():
            spine.set_linewidth(0.45)
            spine.set_edgecolor('#555555')
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / 'attack_pipeline_ghost.pdf', bbox_inches='tight')
    figure.savefig(output / 'attack_pipeline_ghost.png', dpi=300, bbox_inches='tight')
    plt.close(figure)
if __name__ == '__main__':
    arguments = parse_args()
    build(arguments.results, arguments.output)
