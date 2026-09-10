/**
 * Tile geometry for frames that are not square.
 *
 * Imported UNCHANGED by two callers, for the same reason detect.js is:
 *   - the browser demo (site/js/app.js)
 *   - the wide-strip harness (site/verify/strip_check.mjs)
 * One copy of the geometry, two runtimes, one measured comparison.
 *
 * WHY THIS EXISTS
 * The network input is a fixed 256x256 (detect.js -> IMGSZ). Resizing a whole
 * 2048x1000 strip frame into it is an 8.00x squeeze across the width against a 3.91x
 * squeeze down the height: every defect is compressed 2.05x more horizontally than
 * vertically. The model was trained on 200x200 NEU-DET crops resized uniformly, so
 * that anisotropy is a domain shift the weights never saw. A longitudinal scratch
 * becomes 2x thinner relative to its length; the features it fires on move.
 *
 * The fix is not to letterbox -- padding a 2.05:1 frame into a square wastes half the
 * input on grey bars and shrinks the real content to 256x125, which throws away the
 * resolution that made the defect visible. The fix is to cut the frame into SQUARE
 * windows the width of one strip height, resize each uniformly, and merge.
 *
 * GEOMETRY
 *   side  = min(width, height)         a full-height square window
 *   step  <= side * (1 - MIN_OVERLAP)  so a defect on a seam lands whole in a
 *                                      neighbour, instead of being halved in both
 *   n     = the fewest windows that cover the long axis at that step
 *   spread evenly from 0 to L - side, so the first and last are flush with the
 *   frame edges and every interior overlap is identical
 *
 * For 2048x1000 that is 3 windows at x = 0, 524, 1048, each 1000x1000, each resized
 * by a uniform 3.91x. Two would leave a 48 px band uncovered at this side length.
 *
 * Tiles are only worth their extra forward passes when the distortion is real, so a
 * frame inside ASPECT_TOLERANCE of square is left alone and takes the single-pass
 * path -- which is every NEU-DET sample on the page (200x200, ratio 1.000).
 */

/** Squeeze ratio above which the anisotropy is worth extra forward passes. */
export const ASPECT_TOLERANCE = 1.25;

/** Least fraction of a window shared with its neighbour. */
export const MIN_OVERLAP = 0.2;

/**
 * Hard ceiling on windows per frame. A 30:1 panorama would otherwise queue 30
 * sequential WASM forward passes and hang the tab; past this the frame is covered by
 * fewer, wider-stepped windows and the shortfall is reported rather than hidden.
 */
export const MAX_TILES = 12;

/**
 * @param {number} width  frame width in pixels
 * @param {number} height frame height in pixels
 * @param {number} size   network input side (detect.js IMGSZ)
 * @returns {{x:number,y:number,w:number,h:number}[]} windows in frame pixels. Length 1
 *   and covering the whole frame when tiling is not warranted, so callers can always
 *   iterate the result instead of branching.
 */
export function planTiles(width, height, size) {
  const side = Math.min(width, height);
  const long = Math.max(width, height);
  const horizontal = width >= height;

  if (!(long / side >= ASPECT_TOLERANCE) || side < 1) {
    return [{ x: 0, y: 0, w: width, h: height }];
  }

  const step = Math.max(1, side * (1 - MIN_OVERLAP));
  const n = Math.min(MAX_TILES, Math.ceil((long - side) / step) + 1);
  const gap = n > 1 ? (long - side) / (n - 1) : 0;

  const tiles = [];
  for (let i = 0; i < n; i++) {
    // Round to integer pixels: these index an RGBA buffer and a canvas drawImage
    // source rect, neither of which takes a fraction.
    const off = Math.round(i * gap);
    tiles.push(horizontal
      ? { x: off, y: 0, w: side, h: side }
      : { x: 0, y: off, w: side, h: side });
  }
  return tiles;
}

/** True when planTiles would return more than one window. Cheap, for status text. */
export function tilingApplies(width, height) {
  return Math.max(width, height) / Math.max(1, Math.min(width, height)) >= ASPECT_TOLERANCE;
}
