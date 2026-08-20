"""
Magic Recolor Service.

Takes a source image and a painted mask (both as raw bytes), then shifts
the hue/saturation/lightness of the masked region to match a target colour
while preserving the subject's texture and shading.

Algorithm
─────────
1. Decode source image → RGBA (preserves transparent backgrounds).
2. Decode the stroke mask → grayscale (white = painted region).
3. Soft-edge the mask with a Gaussian blur so the recoloured zone blends
   naturally instead of having a hard cut-out edge.
4. Convert the RGB channels of the source to HSV.
5. Compute target hue, saturation, and a *value blend factor* from the
   target colour (e.g. white target → desaturate rather than saturate).
6. For each pixel inside the mask:
      new_hue        = target_hue  (complete override)
      new_saturation = blend(orig_sat, target_sat, mask_alpha)
      new_value      = blend(orig_val, orig_val * value_scale, mask_alpha)
   The value channel is never fully replaced — only scaled — so the
   original lighting / shading is preserved.
7. Re-composite the recoloured RGB back onto the original RGBA frame,
   keeping the alpha channel untouched.
8. Re-encode as PNG bytes and return.

This means:
  • Texture (wrinkles in fabric, hair strands, product embossing) stays.
  • Shadows and highlights remain relative to the original.
  • Semi-transparent edges (hair, fur) are handled correctly.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
import io


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """
    Parse a CSS hex colour string into an (R, G, B) uint8 tuple.

    Accepts: #RGB  #RRGGBB  (leading # optional).
    """
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        raise ValueError(f"Invalid hex colour: {hex_color!r}")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return r, g, b


def _rgb_to_hsv_single(r: int, g: int, b: int) -> tuple[float, float, float]:
    """
    Convert a single uint8 RGB triplet to OpenCV HSV floats.

    Returns (hue ∈ [0, 180), sat ∈ [0, 255], val ∈ [0, 255]) — the
    8-bit scale that cv2.cvtColor uses.
    """
    pixel = np.array([[[b, g, r]]], dtype=np.uint8)          # OpenCV is BGR
    hsv   = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV).astype(np.float32)
    return float(hsv[0, 0, 0]), float(hsv[0, 0, 1]), float(hsv[0, 0, 2])


# ---------------------------------------------------------------------------
# Core recolor
# ---------------------------------------------------------------------------

def recolor_region(
    image_bytes: bytes,
    mask_bytes:  bytes,
    target_hex:  str,
    strength:    float = 1.0,
    feather:     int   = 15,
) -> bytes:
    """
    Recolour the masked region of *image_bytes* to *target_hex*.

    Args:
        image_bytes: Raw source image bytes (JPEG / PNG / WebP).
        mask_bytes:  Grayscale PNG where white marks the painted area.
        target_hex:  Target CSS hex colour (e.g. "#e83c6d").
        strength:    Blend factor 0.0–1.0.  1.0 = full recolour (default).
        feather:     Gaussian blur radius in pixels applied to the mask edge
                     to soften transitions.  0 = hard edge.

    Returns:
        Recoloured image as PNG bytes.

    Raises:
        ValueError: If the hex colour is invalid or the mask cannot be decoded.
    """
    strength = float(max(0.0, min(1.0, strength)))

    # ── 1. Decode source ──────────────────────────────────────────────────
    src_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    src_bgra = cv2.imdecode(src_arr, cv2.IMREAD_UNCHANGED)

    if src_bgra is None:
        raise ValueError("Could not decode source image.")

    has_alpha = src_bgra.ndim == 3 and src_bgra.shape[2] == 4
    if has_alpha:
        src_alpha = src_bgra[:, :, 3].copy()
        src_bgr   = src_bgra[:, :, :3].copy()
    else:
        src_alpha = None
        src_bgr   = src_bgra if src_bgra.ndim == 3 else cv2.cvtColor(src_bgra, cv2.COLOR_GRAY2BGR)

    h, w = src_bgr.shape[:2]

    # ── 2. Decode mask ────────────────────────────────────────────────────
    mask_arr  = np.frombuffer(mask_bytes, dtype=np.uint8)
    mask_gray = cv2.imdecode(mask_arr, cv2.IMREAD_GRAYSCALE)

    if mask_gray is None:
        raise ValueError("Could not decode mask image.")

    # Resize mask to match source in case they differ
    if mask_gray.shape[:2] != (h, w):
        mask_gray = cv2.resize(mask_gray, (w, h), interpolation=cv2.INTER_LINEAR)

    # ── 3. Feather mask edges ─────────────────────────────────────────────
    if feather > 0:
        k = feather * 2 + 1                           # must be odd
        mask_f = cv2.GaussianBlur(
            mask_gray.astype(np.float32),
            (k, k), 0,
        )
    else:
        mask_f = mask_gray.astype(np.float32)

    # Normalise to [0, 1] and scale by strength
    mask_f = (mask_f / 255.0) * strength              # (H, W) float32

    # ── 4. Convert source to HSV ──────────────────────────────────────────
    src_hsv = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    # Channels: H ∈ [0,180), S ∈ [0,255], V ∈ [0,255]

    # ── 5. Target colour in HSV ───────────────────────────────────────────
    tr, tg, tb         = _hex_to_rgb(target_hex)
    t_hue, t_sat, t_val = _rgb_to_hsv_single(tr, tg, tb)

    # ── 6. Apply hue + saturation shift inside the mask ───────────────────
    out_hsv = src_hsv.copy()

    # Expand mask to (H, W, 1) for broadcasting
    alpha_3d = mask_f[:, :, np.newaxis]

    # Hue: full override weighted by mask
    #   new_hue = lerp(orig_hue, target_hue, mask)
    # Note: OpenCV hue wraps at 180 — we rotate via the shortest arc
    orig_hue = src_hsv[:, :, 0]
    hue_diff = t_hue - orig_hue
    # Wrap to [-90, 90) for shortest-arc rotation
    hue_diff = ((hue_diff + 90) % 180) - 90
    new_hue  = (orig_hue + hue_diff * mask_f) % 180
    out_hsv[:, :, 0] = new_hue

    # Saturation: blend original sat toward target sat
    orig_sat = src_hsv[:, :, 1]
    new_sat  = orig_sat + (t_sat - orig_sat) * mask_f
    new_sat  = np.clip(new_sat, 0, 255)
    out_hsv[:, :, 1] = new_sat

    # Value: preserve shading — only scale toward target brightness.
    # We compute how bright the target is relative to a mid-value (128)
    # and nudge the existing value by that factor rather than replacing it.
    # This keeps shadows dark and highlights bright.
    orig_val = src_hsv[:, :, 2]
    val_scale = t_val / 128.0  if t_val > 0 else 0.0   # >1 brightens, <1 darkens
    val_scale = max(0.3, min(2.0, val_scale))           # clamp to avoid extremes
    new_val   = orig_val * (1.0 + (val_scale - 1.0) * mask_f)
    new_val   = np.clip(new_val, 0, 255)
    out_hsv[:, :, 2] = new_val

    # ── 7. Convert back to BGR and re-attach alpha ─────────────────────────
    out_bgr = cv2.cvtColor(out_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if has_alpha:
        out_bgra       = np.dstack([out_bgr, src_alpha])
        encode_img     = out_bgra
        encode_ext     = ".png"
    else:
        encode_img     = out_bgr
        encode_ext     = ".png"

    # ── 8. Encode to PNG bytes ─────────────────────────────────────────────
    success, buf = cv2.imencode(encode_ext, encode_img)
    if not success:
        raise RuntimeError("Failed to encode recoloured image.")
    return buf.tobytes()
