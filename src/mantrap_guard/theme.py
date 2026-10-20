"""Visual theme shared by the per-frame overlay and the showcase: a dark, green-tinted HUD.

Fonts (SIL Open Font License, bundled in ./fonts):
  Rajdhani      - display: titles, labels, big numbers
  Space Grotesk - body text
  JetBrains Mono - terminal
If a font file is missing the loader falls back to common system fonts, then to Pillow's
built-in font, so rendering never fails.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

FONT_DIR = Path(__file__).parent / "fonts"

# --- palette (hex for PIL) ---------------------------------------------------------------
BG = "#030806"  # page background, near-black green
GRID = "#0a1813"  # faint background grid
SURFACE = "#07120e"  # panel fill
SURFACE_2 = "#0b1c16"  # raised fill (tracks of meters, tab strips)
EDGE = "#163b2d"  # panel outline
DEEP_GREEN = "#0f3d2c"  # the signature dark green
ACCENT = "#2fe0a0"  # emerald highlight, used sparingly
ACCENT_DIM = "#1a8c62"
TEXT = "#e8f4ee"
TEXT_2 = "#a3c2b4"
MUTED = "#5b7c6e"
AMBER = "#f0b54a"
RED = "#ff4f5e"
# stage colours: fixed categorical order, validated for colour-vision deficiency on SURFACE
STAGE_COLORS = ["#1aa86e", "#2d8fd2", "#cf7a2c", "#9070dc", "#d0508c"]


def hex_to_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


# --- fonts ------------------------------------------------------------------------------
_FILES = {
    "display": ("Rajdhani-SemiBold.ttf", None),
    "display_medium": ("Rajdhani-Medium.ttf", None),
    "body": ("SpaceGrotesk-Variable.ttf", 400),
    "body_medium": ("SpaceGrotesk-Variable.ttf", 500),
    "body_bold": ("SpaceGrotesk-Variable.ttf", 700),
    "mono": ("JetBrainsMono-Variable.ttf", 400),
    "mono_bold": ("JetBrainsMono-Variable.ttf", 700),
}
_SYSTEM_FALLBACKS = {
    "mono": ["DejaVuSansMono.ttf", "consola.ttf", "Menlo.ttc", "LiberationMono-Regular.ttf"],
    "sans": ["DejaVuSans.ttf", "arial.ttf", "Helvetica.ttc", "LiberationSans-Regular.ttf"],
}
_SYSTEM_DIRS = ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation",
                "/Library/Fonts", "/System/Library/Fonts", "C:/Windows/Fonts"]


@lru_cache(maxsize=64)
def font(role: str, size: int) -> ImageFont.FreeTypeFont:
    """A font for a role ('display', 'body', 'mono', ...) at a pixel size."""
    name, weight = _FILES[role]
    path = FONT_DIR / name
    if path.exists():
        # BASIC layout: ~10x faster than Raqm; the text drawn here is plain ASCII
        f = ImageFont.truetype(str(path), size, layout_engine=ImageFont.Layout.BASIC)
        if weight is not None:
            try:
                f.set_variation_by_axes([weight])
            except Exception:  # FreeType without variable-font support: default weight
                pass
        return f
    for cand in _SYSTEM_FALLBACKS["mono" if role.startswith("mono") else "sans"]:
        for d in _SYSTEM_DIRS:
            if (Path(d) / cand).exists():
                return ImageFont.truetype(str(Path(d) / cand), size,
                                          layout_engine=ImageFont.Layout.BASIC)
    return ImageFont.load_default(size)


# --- fast text for the per-frame overlay --------------------------------------------------
class GlyphCache:
    """Renders each character once per font, then composes strings from the cached glyph
    masks with numpy. Hinted display fonts cost ~2 ms per string through Pillow; composed
    from cached glyphs a label costs a few hundredths of a millisecond, which matters on
    a Jetson CPU that is also running tracking and optical flow."""

    def __init__(self):
        self._g: dict = {}

    def _glyph(self, f, ch):
        key = (id(f), ch)
        g = self._g.get(key)
        if g is None:
            from PIL import Image, ImageDraw
            import numpy as np

            l, t, r, b = f.getbbox(ch)
            w, h = max(1, r - l), max(1, b - t)
            im = Image.new("L", (w, h), 0)
            ImageDraw.Draw(im).text((-l, -t), ch, font=f, fill=255)
            g = (np.asarray(im, np.float32)[..., None] / 255.0, int(l), int(t), f.getlength(ch))
            self._g[key] = g
        return g

    def size(self, text, f):
        """-> (width, height, top offset of the ink from the text origin)"""
        if not text:
            return 0, 0, 0
        gl = [self._glyph(f, c) for c in text]
        top = min(g[2] for g in gl if g[0].shape[0] > 1) if any(g[0].shape[0] > 1 for g in gl) else 0
        bot = max(g[2] + g[0].shape[0] for g in gl)
        return int(round(sum(g[3] for g in gl))), int(bot - top), int(top)

    def draw(self, img, xy, text, f, color_bgr):
        import numpy as np

        x, y = float(xy[0]), int(xy[1])
        col = np.asarray(color_bgr, np.float32)
        H, W = img.shape[:2]
        for ch in text:
            a, l, t, adv = self._glyph(f, ch)
            gx, gy = int(round(x)) + l, y + t
            h, w = a.shape[:2]
            x0, y0, x1, y1 = max(gx, 0), max(gy, 0), min(gx + w, W), min(gy + h, H)
            if x1 > x0 and y1 > y0 and ch != " ":
                aa = a[y0 - gy:y1 - gy, x0 - gx:x1 - gx]
                roi = img[y0:y1, x0:x1]
                roi[:] = (roi * (1 - aa) + col * aa).astype(np.uint8)
            x += adv
