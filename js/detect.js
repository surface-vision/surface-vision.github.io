/**
 * Shared YOLOv8 pre/post-processing for the Jindal Stainless surface-defect model.
 *
 * This module is imported UNCHANGED by two callers:
 *   - the browser demo (site/js/app.js), running onnxruntime-web on WASM
 *   - the parity harness (site/verify/parity.mjs), running onnxruntime-node
 *
 * That is deliberate. The browser demo is only trustworthy if the exact decode and
 * NMS it runs can be executed head to head against the Python ultralytics model on
 * the same images. One copy of the code, two runtimes, one comparison table.
 *
 * Every constant here is fixed by the exported graph and the shipped operating
 * point, not chosen:
 *   input   "images"   float32 [1, 3, 256, 256]  RGB, /255, NCHW
 *   output  "output0"  float32 [1, 10, 1344]
 *   1344 anchors = 32*32 + 16*16 + 8*8   (strides 8, 16, 32 at 256 px)
 *   10 channels  = 4 box (cx, cy, w, h in input pixels) + 6 already-activated
 *                  class scores (no sigmoid, the export bakes it in)
 *
 * Source of the layout: reports/export_summary.json -> verification.graph.
 */

/** Class order is fixed by data/neu-det/data.yaml and by the trained head. */
export const CLASS_NAMES = [
  'crazing',
  'inclusion',
  'patches',
  'pitted_surface',
  'rolled-in_scale',
  'scratches',
];

/** src/inference.py -> CLASS_COLORS, verbatim, so demo and console agree on hue. */
export const CLASS_COLORS = {
  crazing: [255, 130, 0],
  inclusion: [220, 30, 45],
  patches: [55, 150, 255],
  pitted_surface: [170, 80, 220],
  'rolled-in_scale': [240, 190, 0],
  scratches: [0, 200, 160],
};

/** src/inference.py -> DEFAULT_IMGSZ. Chosen on val by src/model_study.py, not inherited. */
export const IMGSZ = 256;

/** reports/operating_point.json -> conf_threshold / iou_threshold. */
export const SHIPPED_CONF = 0.15;
export const NMS_IOU = 0.45;

/** ultralytics utils/nms.py defaults that the shipping predict path uses. */
const MAX_NMS = 30000;
const MAX_DET = 300;
const CLASS_OFFSET = 7680; // ultralytics max_wh, used to make NMS class-aware

/**
 * OpenCV INTER_LINEAR on 8-bit data -- the fixed-point arithmetic, not a float
 * approximation of it. This is exactly what ultralytics' LetterBox does to get a
 * 200x200 NEU-DET frame to the 256x256 network input.
 *
 * Three details are load-bearing and the first draft of this file got two of them
 * wrong, which cost up to 19 px of box position on one test image:
 *
 *  1. Half-pixel centres: src = (dst + 0.5) * scale - 0.5, never dst * scale.
 *  2. The weights are quantised to 11-bit fixed point (2048 levels) BEFORE they are
 *     applied, and the vertical pass combines the two integer rows as
 *       (((b0 * (S0 >> 4)) >> 16) + ((b1 * (S1 >> 4)) >> 16) + 2) >> 2
 *     Doing the same interpolation in float64 and rounding at the end is a different
 *     function: it disagreed with cv2 on 12% of bytes rather than 0.1%, and that was
 *     enough to move marginal boxes and flip one NMS tie.
 *  3. cv2.resize on a uint8 image returns uint8; the /255 happens afterwards. The
 *     network never sees the float interpolant.
 *
 * Measured against cv2.resize(..., INTER_LINEAR) on this host (OpenCV 5.0.0, ARM64):
 * bit-identical on a 2048x1000 -> 256 downscale, and within +/-1 of 255 on 135-210
 * bytes of the 196,608 in a 200x200 -> 256 upscale (0.07-0.11%). Zero is not the
 * right target for the last LSB: OpenCV dispatches this loop to NEON here and to SSE
 * elsewhere, so the reference itself is not bit-portable. +/-1 is, and site/verify
 * measures what +/-1 is worth: 0.0057 of confidence and 0.74 px of box.
 *
 * No padding anywhere. reports/input_pipeline_decision.md measured that lossless
 * padding LOSES 0.024-0.052 mAP50 to plain bilinear scaling at the same tensor size
 * on 8 of 8 configurations, and that nearest-neighbour magnification -- also
 * lossless -- is the worst of five arms. So a non-square frame is scaled (squashed)
 * into the square input rather than letterboxed. For a square source -- every NEU-DET
 * frame -- scaling and letterboxing are the same operation, which is what makes the
 * parity test in site/verify a valid check on this function.
 *
 * @param {Uint8ClampedArray|Uint8Array} src RGBA, srcW*srcH*4
 * @returns {Uint8ClampedArray} RGBA, dstW*dstH*4
 */
export function resizeBilinearRGBA(src, srcW, srcH, dstW, dstH) {
  const X = resizeTaps(dstW, srcW);
  const Y = resizeTaps(dstH, srcH);
  const out = new Uint8ClampedArray(dstW * dstH * 4);

  // Horizontal pass, one source row at a time, into an int buffer at 2048x scale.
  // Two rows are enough: consecutive output rows share at least one source row.
  const rowA = new Int32Array(dstW * 3);
  const rowB = new Int32Array(dstW * 3);
  let haveA = -1;
  let haveB = -1;

  const hresize = (sy, buf) => {
    const base = sy * srcW * 4;
    for (let d = 0; d < dstW; d++) {
      const pa = base + X.ofs[d] * 4;
      const pb = base + X.ofs1[d] * 4;
      const w0 = X.a0[d];
      const w1 = X.a1[d];
      const o = d * 3;
      buf[o] = src[pa] * w0 + src[pb] * w1;
      buf[o + 1] = src[pa + 1] * w0 + src[pb + 1] * w1;
      buf[o + 2] = src[pa + 2] * w0 + src[pb + 2] * w1;
    }
  };

  for (let dy = 0; dy < dstH; dy++) {
    const sy = Y.ofs[dy];
    const sy1 = Y.ofs1[dy];
    if (sy !== haveA) { hresize(sy, rowA); haveA = sy; }
    if (sy1 !== haveB) { hresize(sy1, rowB); haveB = sy1; }
    const b0 = Y.a0[dy];
    const b1 = Y.a1[dy];
    const o = dy * dstW * 4;
    for (let d = 0; d < dstW; d++) {
      const i = d * 3;
      const p = o + d * 4;
      for (let c = 0; c < 3; c++) {
        // OpenCV's VResizeLinear<uchar, int, short, FixedPtCast<int, uchar, 22>>
        const v = (((b0 * (rowA[i + c] >> 4)) >> 16) + ((b1 * (rowB[i + c] >> 4)) >> 16) + 2) >> 2;
        out[p + c] = v;   // Uint8ClampedArray saturates for us
      }
      out[p + 3] = 255;
    }
  }
  return out;
}

/** cvRound: round-half-to-even, which is what saturate_cast<short>(float) does. */
function cvRound(v) {
  const f = Math.floor(v);
  const frac = v - f;
  if (frac > 0.5) return f + 1;
  if (frac < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

/**
 * Source offsets and 11-bit fixed-point weights for one axis, following
 * resize.cpp's setup loop including its border rule: an out-of-range tap collapses
 * onto the edge sample with weight 1, it does not extrapolate.
 */
function resizeTaps(dstN, srcN) {
  const scale = srcN / dstN;
  const ofs = new Int32Array(dstN);
  const ofs1 = new Int32Array(dstN);
  const a0 = new Int32Array(dstN);
  const a1 = new Int32Array(dstN);
  for (let d = 0; d < dstN; d++) {
    // The (float) cast matters: OpenCV computes this in double and stores a float.
    let fx = Math.fround((d + 0.5) * scale - 0.5);
    let sx = Math.floor(fx);
    fx -= sx;
    if (sx < 0) { fx = 0; sx = 0; }
    if (sx >= srcN - 1) { fx = 0; sx = srcN - 1; }
    ofs[d] = sx;
    ofs1[d] = Math.min(sx + 1, srcN - 1); // weight is 0 here whenever it is clamped
    a0[d] = cvRound((1 - fx) * RESIZE_COEF_SCALE);
    a1[d] = cvRound(fx * RESIZE_COEF_SCALE);
  }
  return { ofs, ofs1, a0, a1 };
}

/** OpenCV INTER_RESIZE_COEF_SCALE = 1 << INTER_RESIZE_COEF_BITS, bits = 11. */
const RESIZE_COEF_SCALE = 2048;

/**
 * RGBA at the network input size -> the float32 NCHW tensor the graph wants.
 * RGB channel order, /255, no mean/std normalisation (ultralytics uses none).
 */
export function rgbaToNCHW(rgba, size = IMGSZ) {
  const plane = size * size;
  const t = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    const p = i * 4;
    t[i] = rgba[p] / 255;
    t[plane + i] = rgba[p + 1] / 255;
    t[2 * plane + i] = rgba[p + 2] / 255;
  }
  return t;
}

/** Full preprocess: original-size RGBA in, model tensor out. */
export function preprocess(rgba, srcW, srcH, size = IMGSZ) {
  const resized = resizeBilinearRGBA(rgba, srcW, srcH, size, size);
  return { tensor: rgbaToNCHW(resized, size), resized, size };
}

/**
 * Class-aware greedy NMS, written out rather than borrowed so the browser path has
 * no dependency on it. Matches torchvision.ops.nms as ultralytics calls it:
 * descending score order, strict `iou > threshold` suppression, and boxes offset by
 * class * 7680 so two different defect types never suppress each other.
 *
 * @param {number[][]} boxes xyxy in input-tensor pixels
 * @param {number[]} scores
 * @param {number[]} classes
 * @returns {number[]} kept indices, highest score first
 */
export function nms(boxes, scores, classes, iouThreshold = NMS_IOU) {
  const order = scores
    .map((s, i) => i)
    .sort((a, b) => scores[b] - scores[a])
    .slice(0, MAX_NMS);

  const areas = boxes.map((b) => Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]));
  const suppressed = new Uint8Array(boxes.length);
  const keep = [];

  for (let i = 0; i < order.length; i++) {
    const a = order[i];
    if (suppressed[a]) continue;
    keep.push(a);
    if (keep.length >= MAX_DET) break;
    const [ax1, ay1, ax2, ay2] = boxes[a];
    const off = classes[a] * CLASS_OFFSET;
    for (let j = i + 1; j < order.length; j++) {
      const b = order[j];
      if (suppressed[b]) continue;
      // The class offset makes the two boxes disjoint whenever the classes differ,
      // which is exactly how ultralytics gets class-aware behaviour out of a
      // class-agnostic kernel. Applying it as an early skip is equivalent.
      if (classes[b] !== classes[a]) continue;
      const [bx1, by1, bx2, by2] = boxes[b];
      const iw = Math.min(ax2, bx2) - Math.max(ax1, bx1);
      if (iw <= 0) continue;
      const ih = Math.min(ay2, by2) - Math.max(ay1, by1);
      if (ih <= 0) continue;
      const inter = iw * ih;
      const iou = inter / (areas[a] + areas[b] - inter);
      if (iou > iouThreshold) suppressed[b] = 1;
    }
    void off;
  }
  return keep;
}

/**
 * Decode one raw [1, 10, 1344] output into final detections in ORIGINAL image
 * pixels, in the same order of operations ultralytics uses:
 *
 *   1. transpose to [1344, 10]
 *   2. max over the 6 class columns  (multi_label=False -> best class only)
 *   3. strict `> conf` filter
 *   4. cx,cy,w,h -> x1,y1,x2,y2, still in 256-px input space
 *   5. class-aware NMS at IoU 0.45, still in 256-px input space
 *   6. only then scale to the original frame and clip
 *
 * Step 5 before step 6 matters: for a non-square frame the x and y scales differ,
 * so IoU computed after scaling is not the IoU ultralytics computed.
 *
 * @param {Float32Array} data output0
 * @param {{conf:number, iou:number, origW:number, origH:number, size:number, numAnchors:number}} opt
 */
export function decode(data, opt) {
  const {
    conf = SHIPPED_CONF,
    iou = NMS_IOU,
    origW,
    origH,
    size = IMGSZ,
    numAnchors = data.length / (4 + CLASS_NAMES.length),
  } = opt;

  const nc = CLASS_NAMES.length;
  const stride = numAnchors; // channel-major: data[c * numAnchors + a]

  const boxes = [];
  const scores = [];
  const classes = [];

  for (let a = 0; a < numAnchors; a++) {
    let best = -1;
    let bestCls = -1;
    for (let c = 0; c < nc; c++) {
      const s = data[(4 + c) * stride + a];
      if (s > best) { best = s; bestCls = c; }
    }
    if (!(best > conf)) continue; // strict, as in ultralytics

    const cx = data[a];
    const cy = data[stride + a];
    const w = data[2 * stride + a];
    const h = data[3 * stride + a];
    boxes.push([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]);
    scores.push(best);
    classes.push(bestCls);
  }

  const keep = nms(boxes, scores, classes, iou);
  const sx = origW / size;
  const sy = origH / size;

  return keep.map((i) => {
    const x1 = Math.min(Math.max(boxes[i][0] * sx, 0), origW);
    const y1 = Math.min(Math.max(boxes[i][1] * sy, 0), origH);
    const x2 = Math.min(Math.max(boxes[i][2] * sx, 0), origW);
    const y2 = Math.min(Math.max(boxes[i][3] * sy, 0), origH);
    return {
      classId: classes[i],
      className: CLASS_NAMES[classes[i]],
      raw: scores[i],
      x1, y1, x2, y2,
      width: x2 - x1,
      height: y2 - y1,
      areaFrac: ((x2 - x1) * (y2 - y1)) / (origW * origH),
    };
  });
}
