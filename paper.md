# Bypassing and Hardening Ghost Font: An Anti-AI Typography, Without Machine Learning

**Apostolos Sakellariou**  
Radboud University, Netherlands  
[apostolos.sakellariou@ru.nl](mailto:apostolos.sakellariou@ru.nl)  
July 2026

## Abstract

Ghost Font looks secure because a paused frame shows only a field of dots. A person sees the hidden word when the animation moves. I show that this motion is also the weakness. My program records nine frames, follows the movement shared by most of the background, removes it and colours what remains red. It uses no machine learning and does not need OCR to reveal the letters. On a video carrying “Ghost” as a word, the result is a clear five-letter image with 0.908 overlap against the known text shape. I then use the same attack to test a defense. Instead of moving the whole background in one predictable direction, the hardened version moves small background regions along different circular paths while the letters keep one motion that people can follow. The unchanged decoder can no longer separate the word: overlap falls to 0.076.

For security researchers, hiding information in motion must be tested against video capture and alignment, not only against screenshots.

**Source code:** [github.com/Sakotass/ghost-font-bypass-hardening](https://github.com/Sakotass/ghost-font-bypass-hardening)

**Keywords:** Ghost Font, temporal differencing, motion segmentation, deterministic recognition, defensive design, browser canvas

## Why This Matters

Ghost Font is presented as an “anti-AI font.” In practical terms, it is an animated visual encoding [\[1\]](#references). Foreground and background dots look the same, but they move differently. That difference makes the word visible to a person. Pause the canvas and the word disappears into noise.

That is a strong defense against screenshot reading. It is not a defense against temporal measurement. If a person needs ordered motion to see the letters, an attacker can measure the same order before attempting recognition. The decisive move is therefore not to train or “run better OCR.” It is “remove the motion shared by most of the canvas.”

The decoder does exactly that. It finds the main vertical movement between adjacent frames, lines up the background and keeps the pixels that refuse to disappear. It combines eight comparisons, smooths and cleans the result, then colours the surviving shape red. That red image is already the bypass; turning it into typed text is optional.

I evaluate the idea on a reproduction rather than presenting a private recording as ground truth. The generator fixes the plaintext, seed, motion, resolution and known glyph mask, then measures the encoded video readers receive. The same plaintext, **GHOST**, is used again in the defense so the comparison is direct. Results are reported for this single seeded plaintext as a mechanism demonstration, not as a population success estimate.

**Figure 1 — Measured recovery from `attack/ghost-vulnerable.mp4`.** (a) One frame appears as noise. (b) The next frame is shifted by the measured 4 pixels; cyan/red disagreement exposes minority motion. (c) One aligned subtraction already outlines **GHOST**. (d) Eight residuals reinforce the letter support. (e) Thresholding and component cleaning produce the final red mask. No panel was retouched. [Open the attack-pipeline figure](figures/attack_pipeline_ghost.pdf).

## What the Attack Needs

The attacker has a recording or can read pixels from the displayed canvas. Frames remain ordered and the interval between them is short enough for dot correspondence to survive. The plaintext, source textures, mask and motion parameters are not required by the decoder. A single screenshot is outside scope because it contains no temporal displacement.

The clip is a carrier with the same primitive. Most background dots share one global translation while text dots follow another. This makes every measurement reproducible and gives the experiment an exact reference mask.

The proof is deliberately visual. Figure 1 starts with real decoded video frames and ends with the decoder’s red output. The JSON report records the measured shifts and overlap score. Readers can see whether the recovery worked without trusting an OCR engine.

## Run It in the Browser Console

Open the Ghost Font page: <https://www.mixfont.com/ghost-font>. Open the developer console, paste the contents of [code/ghost_font_browser_decoder.js](https://github.com/Sakotass/ghost-font-bypass-hardening/blob/main/code/ghost_font_browser_decoder.js) and press Enter.

The script finds the visible canvas, captures nine frames, performs the vertical shift search, reconstructs the hidden mask.

## Attack

For every adjacent frame pair, the decoder tries downward shifts from zero through eight pixels. For each candidate it measures mean absolute pixel disagreement inside a margin that excludes frame edges. The shift with the smallest disagreement is retained, because background dots occupy most of the image.

The second frame is aligned using the selected shift and subtracted from the first. Background dots now agree, while dots following the text carrier remain as structured error. The decoder repeats this over nine frames and averages the eight residual maps. Averaging suppresses isolated encoding noise and reinforces motion that repeatedly occupies the same letter support.

The average is smoothed with a Gaussian filter. Otsu’s deterministic threshold separates high residual energy from the remaining background [\[2\]](#references). One closing and one opening consolidate strokes and regions smaller than 500 pixels are removed. The retained pixels are rendered red on white.

### Turn Image into Text

The mask can be transcribed without a learned OCR system. Connected components are ordered from left to right, normalized and compared with fixed A–Z and 0–9 templates using symmetric chamfer distance [\[3\]](#references). This stage contains an explicit alphabet and font-shape prior.

## Results

The vulnerable generator draws foreground and background from independent textures with identical dot statistics. It masks the foreground with **GHOST**, moves the background down by 4 pixels per frame and moves the text texture up by 2 pixels per frame.

The decoder selected a 4-pixel downward displacement for all eight frame pairs. Five components survived cleaning, their union overlapped the known glyph mask by 0.908 IoU and the template stage returned `GHOST`. These values were computed from frames decoded from the released video, not from the lossless source images used to construct it.

| Result                  | Vulnerable |    Hardened |
|:------------------------|-----------:|------------:|
| Mask coverage           |       8.6% |       58.3% |
| Text-mask IoU           |      0.908 |       0.076 |
| Inside/outside residual |      73.31 |       0.996 |
| Template output         |    `GHOST` | `W4 WZWJ W` |
| Expected text recovered |        yes |          no |

**Table 1 — The same attack before and after hardening.**

## Defense

The weakness is not the number of dots, their colour or the typeface. The real problem is that most of the background follows one motion, so the decoder can cancel it with one shift. The defense removes that easy target while keeping a shared motion that still helps a person see the word.

The canvas is divided into 120-pixel tiles. Each background tile follows a circular path with a phase derived from the seed; neighbouring tiles therefore move in different directions at the same instant. All text dots move horizontally as one coherent group. Foreground and background keep the same dot appearance and density distribution. Motion wraps inside each tile so no empty boundary strip reveals the grid in a still frame.

The stronger default uses 55,000 dot centres per texture, a 26-pixel text amplitude and a 36-pixel background amplitude. Its source video is `defense/ghost-hardened.mp4`.

**Figure 2 — Regression test from `defense/ghost-hardened.mp4`.** (a–b) Two decoded frames retain the noise-like still appearance. (c) One global 6-pixel registration cannot cancel independently moving tiles. (d) Eight pairs produce residual energy across the canvas rather than a word-shaped concentration. (e) The unchanged thresholding stage returns a flooded mask. All five panels come from the released video or its decoder output. [Open the defense-pipeline figure](figures/defense_pipeline_ghost.pdf).

The selected shifts were `[6, 6, 7, 6, 6, 6, 1, 6]` pixels. Unlike the vulnerable run, residual energy inside the true text mask was not stronger than the surrounding field: the inside/outside ratio was 0.996. The attack marked 58.3% of the frame, overlap with the known mask fell to 0.076 IoU and the forced template readout became `W4 WZWJ W`. Figure 2 makes the failure visible rather than asking the reader to infer it from a score.

This is a hardening result, not proof of universal machine resistance. It defeats the global vertical-registration decoder tested here. A new attacker could search directly for the smaller group of dots moving together horizontally; this decoder was not implemented or tested in this work. The next step is to test direction-selective filters, local motion estimators, longer recordings, different frame rates and many randomized words.

For deployment as a challenge rather than an artwork, the motion seed should come from a fresh nonce and rotate between attempts. Responses should be bound to that nonce, expire quickly and be rate-limited so an old recording cannot answer a new challenge. An accessible nonvisual alternative remains necessary.

## Limitations

The evaluation contains one seeded plaintext used in two designs. It establishes a mechanism and a regression test. It does not estimate a population success rate. A broader study should randomize message length, font, dot density, motion amplitude, frame interval, resolution and compression, then measure both decoder error and **human reading time** across multiple blinded participants.

The method can also fail when frame order is lost, heavy compression destroys dot correspondence or the dominant background motion exceeds the search range. These are operational boundaries, not evidence that the underlying message is cryptographically protected.

The work is intended for evaluation and improvement of visual security mechanisms.

## Code, Videos and Reproduction

The repository contains live browser decoder, the Python attack, deterministic template reader, vulnerable and hardened generators, both playable videos, measured JSON reports and the exact figure inputs used in this paper:

**GitHub:** [github.com/Sakotass/ghost-font-bypass-hardening](https://github.com/Sakotass/ghost-font-bypass-hardening)

## Conclusion

Ghost Font defeats a static reading strategy, but its readable motion also creates an attack surface. Nine frames are enough to cancel one dominant background carrier and recover **GHOST** as a red image without machine learning. The same exploit then becomes a useful engineering test: when the background is changed from one global translation to independently phased local motion, the decoder floods the frame and stops recovering the plaintext.

The practical lesson is simple. Do not begin with the letters; begin with the motion that hides them. For defense, do not merely add noise; remove the global motion model the attacker can cancel.

## Appendix: Test Environment

The reported run used Python 3.12.13, NumPy 2.3.5, Pillow 12.2.0, SciPy 1.17.0, Matplotlib 3.10.8, FFmpeg 6.1.1 and Node.js 24.14.0. Offline timing, where reported by the scripts, includes video decode, reconstruction, PNG output, template construction and transcription. The repository’s `requirements.txt` records portable minimum package versions rather than pinning this exact workstation.

To rebuild both experiments and figures from the repository root:

```bash
python -m pip install -r requirements.txt
python code/vulnerable_ghost_font_demo.py \
  --output attack/ghost-vulnerable.mp4 \
  --report attack/ghost-attack.json \
  --artifacts attack/artifacts
python code/hardened_ghost_font_demo.py \
  --output defense/ghost-hardened.mp4 \
  --report defense/ghost-defense.json \
  --artifacts defense/artifacts
python code/build_attack_figure.py \
  --results attack --output paper/figures
python code/build_defense_figure.py \
  --results defense --output paper/figures
```

## References

1. E. Lu, “Ghost Font: The Anti-AI Font Only Humans Can Read,” Mixfont, 2026. [Online]. Available: [mixfont.com/ghost-font](https://www.mixfont.com/ghost-font). Accessed: Jul. 2026.
2. N. Otsu, “A threshold selection method from gray-level histograms,” *IEEE Transactions on Systems, Man, and Cybernetics*, vol. 9, no. 1, pp. 62–66, 1979. DOI: [10.1109/TSMC.1979.4310076](https://doi.org/10.1109/TSMC.1979.4310076).
3. G. Borgefors, “Hierarchical chamfer matching: A parametric edge matching algorithm,” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 10, no. 6, pp. 849–865, 1988. DOI: [10.1109/34.9107](https://doi.org/10.1109/34.9107).
