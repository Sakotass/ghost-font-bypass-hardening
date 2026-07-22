from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build the measured hardening figure.')
    parser.add_argument('--results', type=Path, default=Path('defense-test'))
    parser.add_argument('--output', type=Path, default=Path('figures'))
    return parser.parse_args()

def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(0.015, 0.965, label, transform=axis.transAxes, ha='left', va='top', color='white', fontsize=9, fontweight='bold', bbox={'boxstyle': 'square,pad=0.2', 'facecolor': '#222222', 'edgecolor': 'none'})

def build(results: Path, output: Path) -> None:
    artifacts = results / 'artifacts'
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads((results / 'ghost-defense.json').read_text(encoding='utf-8'))
    first = np.asarray(Image.open(artifacts / 'single_frame.png').convert('L'))
    second = np.asarray(Image.open(artifacts / 'next_frame.png').convert('L'))
    pair = np.asarray(Image.open(artifacts / 'registered_pair_residual.png').convert('L'))
    motion = np.asarray(Image.open(artifacts / 'motion_map.png').convert('L'))
    attack = np.asarray(Image.open(artifacts / 'attack_output_full.png').convert('RGB'))
    panels = [(first, 'capture', 'frame 1'), (second, 'advance', 'frame 2'), (pair, 'register', f"global shift: {report['estimated_shifts'][0]} px"), (motion, 'accumulate', 'eight pairs'), (attack, 'attack result', 'flooded mask')]
    figure, axes = plt.subplots(1, 5, figsize=(9.4, 2.05), constrained_layout=True)
    for index, (axis, (image, title, subtitle)) in enumerate(zip(axes, panels)):
        axis.imshow(image, cmap='gray', vmin=0, vmax=255)
        axis.set_title(title, fontsize=8, fontweight='bold', pad=2)
        axis.set_xlabel(subtitle, fontsize=6.5, labelpad=2)
        panel_label(axis, chr(ord('a') + index))
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color('#777777')
            spine.set_linewidth(0.55)
    pdf = output / 'defense_pipeline_ghost.pdf'
    png = output / 'defense_pipeline_ghost.png'
    figure.savefig(pdf, bbox_inches='tight')
    figure.savefig(png, dpi=210, bbox_inches='tight')
    plt.close(figure)
if __name__ == '__main__':
    arguments = parse_args()
    build(arguments.results, arguments.output)
