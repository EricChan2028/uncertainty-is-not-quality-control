"""Layout QA for matplotlib schematic figures.

check(fig, ax, scale, min_pt) returns a list of human-readable problems:
  * font: any text (including math sub/superscripts) printed below min_pt at the given print scale
  * text-text: two texts whose boxes intersect
  * text-graphic: a text box hit by a foreground graphic (tokens, icons, arrows, lines), sampled along
    the true path (Bezier curves included), unless the text is a label sitting inside that graphic
  * overflow: a text that sticks out of the smallest background card that contains its centre
"""
import re

import numpy as np
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch
from matplotlib.transforms import Bbox


def _path_points(artist):
    if isinstance(artist, Line2D):
        xy = artist.get_transform().transform(artist.get_xydata())
        pts = [xy[0:1]]
        for a, b in zip(xy[:-1], xy[1:]):
            n = max(2, int(np.hypot(*(b - a)) / 1.0))
            pts.append(np.linspace(a, b, n))
        return np.concatenate(pts) if len(xy) else np.zeros((0, 2))
    path = artist.get_path()
    trans = artist.get_transform()
    pts = []
    for bez, _code in path.iter_bezier():
        ctrl = trans.transform(bez.control_points)
        length = np.sum(np.hypot(*np.diff(ctrl, axis=0).T)) if len(ctrl) > 1 else 0.0
        n = max(2, int(length / 1.0))
        t = np.linspace(0, 1, n)
        from matplotlib.bezier import BezierSegment
        pts.append(BezierSegment(ctrl).point_at_t(t))
    return np.concatenate(pts) if pts else np.zeros((0, 2))


def _shrink(bb, px):
    return Bbox.from_extents(bb.x0 + px, bb.y0 + px, bb.x1 - px, bb.y1 - px)


def _inter_area(a, b):
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return w * h if (w > 0 and h > 0) else 0.0


def _contains(outer, inner, tol=1.0):
    return (inner.x0 >= outer.x0 - tol and inner.y0 >= outer.y0 - tol and
            inner.x1 <= outer.x1 + tol and inner.y1 <= outer.y1 + tol)


def _desc(t):
    return repr(t.get_text().replace("\n", " ")[:40])


def check(fig, ax, scale=1.0, min_pt=6.0, big_in=0.9, ignore_text=()):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    dpi = fig.dpi
    problems = []

    texts = [t for t in ax.texts if t.get_visible() and t.get_text().strip() and t.get_text() not in ignore_text]

    # fonts
    for t in texts:
        s = t.get_text()
        f = 1.0
        if "$" in s:
            if re.search(r"\^\{[^}]*\^", s) or re.search(r"_\{[^}]*_", s):
                f = 0.49
            elif "_" in s or "^" in s or r"\frac" in s:
                f = 0.7
        eff = t.get_fontsize() * f * scale
        if eff < min_pt - 1e-6:
            problems.append(f"font {eff:.2f}pt < {min_pt}pt: {_desc(t)}")

    tb = {t: t.get_window_extent(renderer=r) for t in texts}

    # classify artists
    fg, bg = [], []
    for a in list(ax.patches) + list(ax.lines):
        if not a.get_visible():
            continue
        alpha = a.get_alpha()
        if isinstance(a, Patch) and alpha is not None and alpha < 0.3:
            continue  # shadows
        if isinstance(a, (FancyArrowPatch, Line2D)):
            fg.append(a)
            continue
        bb = a.get_window_extent(renderer=r)
        if max(bb.width, bb.height) / dpi > big_in:
            bg.append((a, bb))
        else:
            fg.append(a)

    # text-text
    for i, a in enumerate(texts):
        for b in texts[i + 1:]:
            if _inter_area(_shrink(tb[a], 0.8), _shrink(tb[b], 0.8)) > 1.0:
                problems.append(f"text-text: {_desc(a)} x {_desc(b)}")

    # text-graphic
    cache = {}
    for a in fg:
        bb = a.get_window_extent(renderer=r)
        pts = _path_points(a)
        cache[a] = (bb, pts)
    for t in texts:
        box = _shrink(tb[t], 1.0)
        own = t.get_bbox_patch()
        for a in fg:
            if a is own:
                continue
            bb, pts = cache[a]
            if _inter_area(box, bb) <= 0:
                continue
            # label sitting inside a filled glyph (token, badge, document) is intended
            if isinstance(a, Patch) and not isinstance(a, FancyArrowPatch) and a.get_fill() and _contains(bb, tb[t], tol=1.5):
                continue
            if len(pts):
                inside = ((pts[:, 0] > box.x0) & (pts[:, 0] < box.x1) & (pts[:, 1] > box.y0) & (pts[:, 1] < box.y1))
                hit = inside.any()
                # a filled patch that covers the text box also counts, even if its outline misses it
                if not hit and isinstance(a, Patch) and a.get_fill() and not isinstance(a, FancyArrowPatch):
                    hit = _inter_area(box, bb) > 0.25 * box.width * box.height and a.contains_point(
                        ((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2))
            else:
                hit = False
            if hit:
                problems.append(f"text-graphic: {_desc(t)} hit by {type(a).__name__}"
                                f" at ({bb.x0 / dpi:.2f},{bb.y0 / dpi:.2f})in")

    # overflow out of the smallest containing background card
    for t in texts:
        c = tb[t]
        cx, cy = (c.x0 + c.x1) / 2, (c.y0 + c.y1) / 2
        conts = [(a, bb) for a, bb in bg if bb.x0 <= cx <= bb.x1 and bb.y0 <= cy <= bb.y1]
        if not conts:
            continue
        a, bb = min(conts, key=lambda z: z[1].width * z[1].height)
        if not _contains(bb, c, tol=0.5):
            problems.append(f"overflow: {_desc(t)} leaves its card")
    return problems


def arrow_report(ax):
    """Describe each curved arc3 arrow: from, to, and the side it bulges toward."""
    out = []
    for a in ax.patches:
        if isinstance(a, FancyArrowPatch) and getattr(a, "_bulge_note", None):
            out.append(a._bulge_note)
    return out
