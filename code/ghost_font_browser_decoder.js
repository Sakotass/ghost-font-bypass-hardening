(async () => {
  "use strict";

  const CONFIG = {
    frameCount: 9,
    intervalMs: 40,
    maxDownwardShift: 12,
    maxProcessingWidth: 1280,
    blurRadius: 4,
    minAreaAt1280x720: 500,
    padding: 40,
    red: [230, 0, 0],
  };

  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  function isVisible(element) {
    const rect = element.getBoundingClientRect();
    const style = getComputedStyle(element);
    return (
      rect.width > 20 &&
      rect.height > 20 &&
      style.display !== "none" &&
      style.visibility !== "hidden" &&
      Number(style.opacity || 1) > 0
    );
  }

  function findTarget() {
    if (
      window.GHOST_FONT_TARGET instanceof HTMLCanvasElement ||
      window.GHOST_FONT_TARGET instanceof HTMLVideoElement
    ) {
      return window.GHOST_FONT_TARGET;
    }

    const mixfontCanvas = document.querySelector(
      'canvas[aria-label="Ghost Font animation preview"], canvas.GhostFontLanding_canvas__lW_ad'
    );
    if (mixfontCanvas && isVisible(mixfontCanvas)) return mixfontCanvas;

    const candidates = [...document.querySelectorAll("canvas, video")].filter(isVisible);
    candidates.sort((a, b) => {
      const ar = a.getBoundingClientRect();
      const br = b.getBoundingClientRect();
      return br.width * br.height - ar.width * ar.height;
    });

    if (!candidates.length) {
      throw new Error("No visible <canvas> or <video> was found on this page.");
    }
    return candidates[0];
  }

  function sourceDimensions(target) {
    const rect = target.getBoundingClientRect();
    if (target instanceof HTMLVideoElement) {
      return {
        width: target.videoWidth || Math.round(rect.width),
        height: target.videoHeight || Math.round(rect.height),
      };
    }
    return {
      width: target.width || Math.round(rect.width * devicePixelRatio),
      height: target.height || Math.round(rect.height * devicePixelRatio),
    };
  }

  function captureGray(target, context, width, height) {
    context.clearRect(0, 0, width, height);
    context.drawImage(target, 0, 0, width, height);
    const rgba = context.getImageData(0, 0, width, height).data;
    const gray = new Uint8Array(width * height);

    for (let src = 0, dst = 0; dst < gray.length; src += 4, dst += 1) {
      gray[dst] = (rgba[src] * 77 + rgba[src + 1] * 150 + rgba[src + 2] * 29) >> 8;
    }
    return gray;
  }

  function bestDownwardShift(a, b, width, height, maxShift) {
    const marginY = Math.max(10, maxShift + 2);
    const marginX = 10;
    const sampleStep = 4;
    let bestShift = 0;
    let bestError = Infinity;

    for (let dy = 0; dy <= maxShift; dy += 1) {
      let error = 0;
      let samples = 0;
      const stopY = height - marginY - dy;

      for (let y = marginY; y < stopY; y += sampleStep) {
        const rowA = y * width;
        const rowB = (y + dy) * width;
        for (let x = marginX; x < width - marginX; x += sampleStep) {
          error += Math.abs(a[rowA + x] - b[rowB + x]);
          samples += 1;
        }
      }

      error /= Math.max(samples, 1);
      if (error < bestError) {
        bestError = error;
        bestShift = dy;
      }
    }
    return bestShift;
  }

  function boxBlur(source, width, height, radius) {
    if (radius <= 0) return new Float32Array(source);
    const horizontal = new Float32Array(source.length);
    const output = new Float32Array(source.length);

    for (let y = 0; y < height; y += 1) {
      const row = y * width;
      let sum = 0;
      let count = Math.min(width, radius + 1);
      for (let x = 0; x < count; x += 1) sum += source[row + x];
      for (let x = 0; x < width; x += 1) {
        if (x > 0) {
          const entering = x + radius;
          const leaving = x - radius - 1;
          if (entering < width) {
            sum += source[row + entering];
            count += 1;
          }
          if (leaving >= 0) {
            sum -= source[row + leaving];
            count -= 1;
          }
        }
        horizontal[row + x] = sum / Math.max(count, 1);
      }
    }

    for (let x = 0; x < width; x += 1) {
      let sum = 0;
      let count = Math.min(height, radius + 1);
      for (let y = 0; y < count; y += 1) sum += horizontal[y * width + x];
      for (let y = 0; y < height; y += 1) {
        if (y > 0) {
          const entering = y + radius;
          const leaving = y - radius - 1;
          if (entering < height) {
            sum += horizontal[entering * width + x];
            count += 1;
          }
          if (leaving >= 0) {
            sum -= horizontal[leaving * width + x];
            count -= 1;
          }
        }
        output[y * width + x] = sum / Math.max(count, 1);
      }
    }
    return output;
  }

  function otsuThreshold(values, width, height, margin) {
    const histogram = new Float64Array(256);
    let total = 0;
    let totalSum = 0;

    for (let y = margin; y < height - margin; y += 1) {
      const row = y * width;
      for (let x = margin; x < width - margin; x += 1) {
        const value = Math.max(0, Math.min(255, Math.round(values[row + x])));
        histogram[value] += 1;
        total += 1;
        totalSum += value;
      }
    }

    let backgroundWeight = 0;
    let backgroundSum = 0;
    let bestThreshold = 0;
    let bestVariance = -1;

    for (let threshold = 0; threshold < 256; threshold += 1) {
      backgroundWeight += histogram[threshold];
      if (!backgroundWeight) continue;
      const foregroundWeight = total - backgroundWeight;
      if (!foregroundWeight) break;

      backgroundSum += threshold * histogram[threshold];
      const backgroundMean = backgroundSum / backgroundWeight;
      const foregroundMean = (totalSum - backgroundSum) / foregroundWeight;
      const variance =
        backgroundWeight *
        foregroundWeight *
        (backgroundMean - foregroundMean) ** 2;

      if (variance > bestVariance) {
        bestVariance = variance;
        bestThreshold = threshold;
      }
    }
    return bestThreshold;
  }

  function dilate(mask, width, height) {
    const output = new Uint8Array(mask.length);
    for (let y = 1; y < height - 1; y += 1) {
      for (let x = 1; x < width - 1; x += 1) {
        const i = y * width + x;
        output[i] =
          mask[i] ||
          mask[i - 1] ||
          mask[i + 1] ||
          mask[i - width] ||
          mask[i + width]
            ? 1
            : 0;
      }
    }
    return output;
  }

  function erode(mask, width, height) {
    const output = new Uint8Array(mask.length);
    for (let y = 1; y < height - 1; y += 1) {
      for (let x = 1; x < width - 1; x += 1) {
        const i = y * width + x;
        output[i] =
          mask[i] &&
          mask[i - 1] &&
          mask[i + 1] &&
          mask[i - width] &&
          mask[i + width]
            ? 1
            : 0;
      }
    }
    return output;
  }

  function removeSmallComponents(mask, width, height, minimumArea) {
    const visited = new Uint8Array(mask.length);
    const stack = new Int32Array(mask.length);

    for (let start = 0; start < mask.length; start += 1) {
      if (!mask[start] || visited[start]) continue;

      let head = 0;
      let size = 1;
      stack[0] = start;
      visited[start] = 1;

      while (head < size) {
        const current = stack[head++];
        const x = current % width;
        const y = (current / width) | 0;
        let next;
        if (x > 0) {
          next = current - 1;
          if (mask[next] && !visited[next]) {
            visited[next] = 1;
            stack[size++] = next;
          }
        }
        if (x + 1 < width) {
          next = current + 1;
          if (mask[next] && !visited[next]) {
            visited[next] = 1;
            stack[size++] = next;
          }
        }
        if (y > 0) {
          next = current - width;
          if (mask[next] && !visited[next]) {
            visited[next] = 1;
            stack[size++] = next;
          }
        }
        if (y + 1 < height) {
          next = current + width;
          if (mask[next] && !visited[next]) {
            visited[next] = 1;
            stack[size++] = next;
          }
        }
      }

      if (size < minimumArea) {
        for (let i = 0; i < size; i += 1) mask[stack[i]] = 0;
      }
    }
  }

  function reconstruct(frames, width, height) {
    const residualSum = new Float32Array(width * height);
    const residualCount = new Uint8Array(width * height);

    for (let pair = 0; pair < frames.length - 1; pair += 1) {
      const a = frames[pair];
      const b = frames[pair + 1];
      const dy = bestDownwardShift(
        a,
        b,
        width,
        height,
        CONFIG.maxDownwardShift
      );
      const usableHeight = height - dy;

      for (let y = 0; y < usableHeight; y += 1) {
        const rowA = y * width;
        const rowB = (y + dy) * width;
        for (let x = 0; x < width; x += 1) {
          const index = rowA + x;
          residualSum[index] += Math.abs(a[index] - b[rowB + x]);
          residualCount[index] += 1;
        }
      }
    }

    const meanResidual = new Float32Array(residualSum.length);
    for (let i = 0; i < meanResidual.length; i += 1) {
      meanResidual[i] = residualSum[i] / Math.max(residualCount[i], 1);
    }

    let motion = boxBlur(meanResidual, width, height, CONFIG.blurRadius);
    motion = boxBlur(motion, width, height, CONFIG.blurRadius);

    const margin = Math.max(20, CONFIG.maxDownwardShift + 5);
    const threshold = otsuThreshold(motion, width, height, margin);
    let mask = new Uint8Array(width * height);

    for (let y = margin; y < height - margin; y += 1) {
      const row = y * width;
      for (let x = margin; x < width - margin; x += 1) {
        mask[row + x] = motion[row + x] > threshold ? 1 : 0;
      }
    }

    mask = erode(dilate(mask, width, height), width, height);
    mask = dilate(erode(mask, width, height), width, height);

    const scale = (width * height) / (1280 * 720);
    const minimumArea = Math.max(80, Math.round(CONFIG.minAreaAt1280x720 * scale));
    removeSmallComponents(mask, width, height, minimumArea);
    return mask;
  }

  function renderResult(mask, width, height) {
    let minX = width;
    let minY = height;
    let maxX = -1;
    let maxY = -1;

    const full = document.createElement("canvas");
    full.width = width;
    full.height = height;
    const context = full.getContext("2d", { willReadFrequently: true });
    const pixels = context.createImageData(width, height);

    for (let i = 0; i < mask.length; i += 1) {
      const offset = i * 4;
      const foreground = Boolean(mask[i]);
      pixels.data[offset] = foreground ? CONFIG.red[0] : 255;
      pixels.data[offset + 1] = foreground ? CONFIG.red[1] : 255;
      pixels.data[offset + 2] = foreground ? CONFIG.red[2] : 255;
      pixels.data[offset + 3] = 255;

      if (foreground) {
        const x = i % width;
        const y = (i / width) | 0;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }

    if (maxX < 0) {
      throw new Error("No foreground was recovered. Let the animation play and run again.");
    }
    context.putImageData(pixels, 0, 0);

    minX = Math.max(0, minX - CONFIG.padding);
    minY = Math.max(0, minY - CONFIG.padding);
    maxX = Math.min(width - 1, maxX + CONFIG.padding);
    maxY = Math.min(height - 1, maxY + CONFIG.padding);

    const cropped = document.createElement("canvas");
    cropped.width = maxX - minX + 1;
    cropped.height = maxY - minY + 1;
    cropped
      .getContext("2d")
      .drawImage(
        full,
        minX,
        minY,
        cropped.width,
        cropped.height,
        0,
        0,
        cropped.width,
        cropped.height
      );
    return cropped;
  }

  function showOverlay(resultCanvas, target) {
    document.getElementById("ghost-font-decoder-result")?.remove();

    const panel = document.createElement("div");
    panel.id = "ghost-font-decoder-result";
    Object.assign(panel.style, {
      position: "fixed",
      inset: "20px",
      zIndex: "2147483647",
      padding: "18px",
      background: "rgba(15, 15, 18, 0.97)",
      border: "1px solid #555",
      borderRadius: "12px",
      boxShadow: "0 20px 70px rgba(0,0,0,.6)",
      color: "white",
      font: "14px/1.4 system-ui, sans-serif",
      overflow: "auto",
      textAlign: "center",
    });

    const title = document.createElement("div");
    title.textContent = `Recovered motion mask from ${target.tagName.toLowerCase()}`;
    title.style.marginBottom = "12px";
    title.style.fontWeight = "700";

    Object.assign(resultCanvas.style, {
      display: "block",
      maxWidth: "100%",
      height: "auto",
      margin: "0 auto 14px",
      background: "white",
      borderRadius: "6px",
    });

    const download = document.createElement("button");
    download.textContent = "Download PNG";
    const close = document.createElement("button");
    close.textContent = "Close";

    for (const button of [download, close]) {
      Object.assign(button.style, {
        margin: "0 6px",
        padding: "9px 14px",
        border: "0",
        borderRadius: "7px",
        cursor: "pointer",
        font: "600 14px system-ui, sans-serif",
      });
    }

    download.onclick = () => {
      resultCanvas.toBlob((blob) => {
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = "ghost-font-recovered.png";
        link.click();
        setTimeout(() => URL.revokeObjectURL(link.href), 1000);
      }, "image/png");
    };
    close.onclick = () => panel.remove();

    panel.append(title, resultCanvas, download, close);
    document.documentElement.appendChild(panel);
  }

  try {
    const target = findTarget();
    if (target instanceof HTMLVideoElement && target.paused) {
      await target.play().catch(() => {});
    }

    const source = sourceDimensions(target);
    const scale = Math.min(1, CONFIG.maxProcessingWidth / source.width);
    const width = Math.max(1, Math.round(source.width * scale));
    const height = Math.max(1, Math.round(source.height * scale));

    const captureCanvas = document.createElement("canvas");
    captureCanvas.width = width;
    captureCanvas.height = height;
    const captureContext = captureCanvas.getContext("2d", {
      willReadFrequently: true,
    });

    console.log(
      `[Ghost decoder] Capturing ${CONFIG.frameCount} frames from`,
      target
    );
    const frames = [];
    for (let i = 0; i < CONFIG.frameCount; i += 1) {
      frames.push(captureGray(target, captureContext, width, height));
      if (i + 1 < CONFIG.frameCount) await sleep(CONFIG.intervalMs);
    }

    const mask = reconstruct(frames, width, height);
    const result = renderResult(mask, width, height);
    showOverlay(result, target);
    console.log("[Ghost decoder] Reconstruction complete.");
  } catch (error) {
    console.error("[Ghost decoder]", error);
    const crossOrigin =
      error?.name === "SecurityError" || /taint|cross-origin/i.test(error?.message || "");
    alert(
      crossOrigin
        ? "The browser blocked pixel access because the canvas/video is cross-origin. A screen-capture version is required for this page."
        : `Ghost decoder failed: ${error?.message || error}`
    );
  }
})();
