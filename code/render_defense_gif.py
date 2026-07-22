from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from hardened_ghost_font_demo import generate_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the hardened Ghost Font defense as a smooth, looping GIF."
    )
    parser.add_argument("--text", default="GHOST")
    parser.add_argument(
        "--output", type=Path, default=Path("defense/ghost-hardened.gif")
    )
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--display-width", type=int, default=480)
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--fps", type=int, default=18)
    parser.add_argument("--dots", type=int, default=30_000)
    parser.add_argument("--tile-size", type=int, default=90)
    parser.add_argument("--amplitude", type=float, default=20.0)
    parser.add_argument("--background-amplitude", type=float, default=27.0)
    parser.add_argument("--seed", type=int, default=1_195_921_235)
    return parser.parse_args()


def encode_gif(frames: Path, output: Path, fps: int, display_width: int) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found on PATH")
    output.parent.mkdir(parents=True, exist_ok=True)
    filters = (
        f"fps={fps},scale={display_width}:-1:flags=lanczos,split[a][b];"
        "[a]palettegen=max_colors=96:stats_mode=diff[p];"
        "[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frames / "frame_%06d.png"),
            "-filter_complex",
            filters,
            "-loop",
            "0",
            str(output),
        ],
        check=True,
    )


def main() -> None:
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="ghost-defense-gif-") as temp:
        frame_dir = Path(temp)
        generate_frames(
            frame_dir,
            text=args.text,
            width=args.width,
            height=args.height,
            count=args.frames,
            fps=args.fps,
            dot_count=args.dots,
            tile_size=args.tile_size,
            amplitude=args.amplitude,
            background_amplitude=args.background_amplitude,
            seed=args.seed,
            text_motion="horizontal",
            background_motion="circular",
        )
        encode_gif(frame_dir, args.output, args.fps, args.display_width)


if __name__ == "__main__":
    main()
