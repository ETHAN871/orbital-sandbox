#!/usr/bin/env python3
"""Generate visual mockups of candidate field-visualization styles.

Run: python tests/fieldviz_mockups.py
Outputs: tests/mockup_3d.png, tests/mockup_2d.png, tests/mockup_flow.png

Used as a discussion aid — these are NOT the actual app rendering, they
are intent-of-design mockups so the user can pick a direction.
"""

import math
import random
import os
from PIL import Image, ImageDraw, ImageFont

W, H = 1200, 800
BG = (10, 10, 15)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Two bodies that all mockups share — same scene, three different field viz.
BODY_A = {"x": 460, "y": 430, "r": 38, "m": 200, "color": (255, 150, 90)}     # orange, mass 200
BODY_B = {"x": 820, "y": 430, "r": 24, "m": 100, "color": (130, 200, 255)}    # cyan, mass 100
G = 1.0          # mock gravitational constant for visualization
EPS = 12.0        # softening so vertices don't pin onto body center

def _try_font(names, size):
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except Exception:
            pass
    return ImageFont.load_default()

FONT_TITLE = _try_font(["arial.ttf", "DejaVuSans.ttf"], 26)
FONT_CAPTION = _try_font(["arial.ttf", "DejaVuSans.ttf"], 16)
FONT_LABEL = _try_font(["arial.ttf", "DejaVuSans.ttf"], 14)

def _bodies(im):
    """Draw the two reference bodies (shared across all mockups)."""
    draw = ImageDraw.Draw(im, "RGBA")
    for b in (BODY_A, BODY_B):
        # Soft outer glow — concentric translucent rings.
        for i in range(6, 0, -1):
            r = b["r"] + i * 4
            alpha = max(0, 30 - i * 4)
            c = (*b["color"], alpha)
            draw.ellipse([b["x"] - r, b["y"] - r, b["x"] + r, b["y"] + r], fill=c)
        # Solid body.
        draw.ellipse(
            [b["x"] - b["r"], b["y"] - b["r"], b["x"] + b["r"], b["y"] + b["r"]],
            fill=(*b["color"], 255),
            outline=(255, 255, 255, 90), width=1,
        )
        label = f"M = {b['m']}"
        bbox = draw.textbbox((0, 0), label, font=FONT_LABEL)
        tw = bbox[2] - bbox[0]
        # Place label ABOVE body so the field viz below stays visible
        # and the label doesn't get washed out by the body glow.
        draw.text(
            (b["x"] - tw // 2, b["y"] - b["r"] - 28),
            label,
            font=FONT_LABEL,
            fill=(230, 230, 240, 230),
        )

def _header(im, title, caption):
    draw = ImageDraw.Draw(im, "RGBA")
    draw.text((40, 28), title, font=FONT_TITLE, fill=(230, 235, 240))
    draw.text((40, 62), caption, font=FONT_CAPTION, fill=(140, 160, 185))
    draw.line([(40, 100), (W - 40, 100)], fill=(60, 70, 90, 140), width=1)

def _phi(x, y):
    """Gravitational potential at (x, y) summed over both bodies."""
    p = 0.0
    for b in (BODY_A, BODY_B):
        dx = x - b["x"]
        dy = y - b["y"]
        r = math.sqrt(dx * dx + dy * dy + EPS * EPS)
        p += -b["m"] / r
    return p

def _grad_phi(x, y):
    """Gradient of phi — points AWAY from attractor."""
    gx, gy = 0.0, 0.0
    for b in (BODY_A, BODY_B):
        dx = x - b["x"]
        dy = y - b["y"]
        r2 = dx * dx + dy * dy + EPS * EPS
        invR3 = 1.0 / (r2 * math.sqrt(r2))
        gx += b["m"] * dx * invR3
        gy += b["m"] * dy * invR3
    return gx, gy


def _local_depth(x, y):
    """Localized depression per body — wells are dramatic near each
    body and decay smoothly. Tuned for visible-but-not-overwhelming."""
    d = 0.0
    for b in (BODY_A, BODY_B):
        dx = x - b["x"]
        dy = y - b["y"]
        r2 = dx * dx + dy * dy + 25.0 * 25.0    # softening
        r = math.sqrt(r2)
        # Mass-weighted depression with Gaussian-like falloff.
        falloff = math.exp(-r / 180.0)
        d += b["m"] * falloff * (1.0 / r)
    return d

def mockup_3d():
    im = Image.new("RGBA", (W, H), BG + (255,))
    draw = ImageDraw.Draw(im, "RGBA")
    cols, rows = 70, 42
    Y_TOP, Y_BOTTOM = 130, 720
    X_LEFT, X_RIGHT = 60, W - 60
    cw = (X_RIGHT - X_LEFT) / (cols - 1)
    rh = (Y_BOTTOM - Y_TOP) / (rows - 1)
    pts = [[(0.0, 0.0)] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            x = X_LEFT + c * cw
            y = Y_TOP + r * rh
            depth = _local_depth(x, y) * 60.0
            sy = y + depth
            pts[r][c] = (x, sy)
    # Draw both sets of lines with depth-modulated alpha — deeper rows
    # are slightly more opaque, hinting at "closer/below" in 3D.
    for r in range(rows):
        for c in range(cols - 1):
            (x1, y1), (x2, y2) = pts[r][c], pts[r][c + 1]
            avgy = (y1 + y2) * 0.5
            # alpha modulation: rows pulled lower get brighter
            base_y = Y_TOP + r * rh
            sag = avgy - base_y
            alpha = 110 + min(80, int(sag * 0.6))
            draw.line([(x1, y1), (x2, y2)], fill=(155, 195, 240, alpha), width=1)
    for c in range(cols):
        for r in range(rows - 1):
            (x1, y1), (x2, y2) = pts[r][c], pts[r + 1][c]
            avgy = (y1 + y2) * 0.5
            base_y = Y_TOP + r * rh
            sag = avgy - base_y
            alpha = 110 + min(80, int(sag * 0.6))
            draw.line([(x1, y1), (x2, y2)], fill=(155, 195, 240, alpha), width=1)
    _bodies(im)
    _header(
        im,
        "3D Spacetime Fabric",
        "rubber-sheet style · grid sinks below masses · classical GR mental model",
    )
    path = os.path.join(OUT_DIR, "mockup_3d.png")
    im.save(path)
    return path


def mockup_2d():
    im = Image.new("RGBA", (W, H), BG + (255,))
    draw = ImageDraw.Draw(im, "RGBA")
    cols, rows = 60, 36
    Y_TOP, Y_BOTTOM = 130, 760
    X_LEFT, X_RIGHT = 60, W - 60
    cw = (X_RIGHT - X_LEFT) / (cols - 1)
    rh = (Y_BOTTOM - Y_TOP) / (rows - 1)
    pts = [[(0.0, 0.0)] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            x = X_LEFT + c * cw
            y = Y_TOP + r * rh
            gx, gy = _grad_phi(x, y)
            scale = 14000.0
            dx = -gx * scale
            dy = -gy * scale
            mag = math.sqrt(dx * dx + dy * dy)
            cap = 32.0
            if mag > cap:
                dx *= cap / mag
                dy *= cap / mag
            pts[r][c] = (x + dx, y + dy)
    for r in range(rows):
        for c in range(cols - 1):
            (x1, y1), (x2, y2) = pts[r][c], pts[r][c + 1]
            draw.line([(x1, y1), (x2, y2)], fill=(160, 200, 250, 140), width=1)
    for c in range(cols):
        for r in range(rows - 1):
            (x1, y1), (x2, y2) = pts[r][c], pts[r + 1][c]
            draw.line([(x1, y1), (x2, y2)], fill=(160, 200, 250, 140), width=1)
    _bodies(im)
    _header(
        im,
        "2D In-plane Warp",
        "flat top-down · grid lines bend toward masses · cells compress near bodies",
    )
    path = os.path.join(OUT_DIR, "mockup_2d.png")
    im.save(path)
    return path


def mockup_flow():
    from PIL import ImageFilter
    im = Image.new("RGBA", (W, H), (4, 5, 12, 255))
    # Two layered glow buffers — bright core + soft halo — for true
    # luminous streamlines.
    core = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    halo = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    cdraw = ImageDraw.Draw(core, "RGBA")
    hdraw = ImageDraw.Draw(halo, "RGBA")
    rng = random.Random(42)
    NUM_PARTICLES = 6000
    TRAIL_STEPS = 28
    STEP = 1.8
    for _ in range(NUM_PARTICLES):
        for _try in range(8):
            x = rng.uniform(20, W - 20)
            y = rng.uniform(115, H - 30)
            d1 = math.hypot(x - BODY_A["x"], y - BODY_A["y"])
            d2 = math.hypot(x - BODY_B["x"], y - BODY_B["y"])
            if d1 > BODY_A["r"] + 25 and d2 > BODY_B["r"] + 25:
                break
        # Tangential bias — some particles orbit rather than free-fall.
        # Sample (vx,vy) perpendicular to nearest body radius.
        nearest = BODY_A if d1 < d2 else BODY_B
        rx = x - nearest["x"]
        ry = y - nearest["y"]
        rmag = math.hypot(rx, ry) + 1e-6
        tx, ty = -ry / rmag, rx / rmag       # tangent (CCW)
        tan_bias = rng.uniform(-1.2, 1.2)
        vx = tan_bias * tx
        vy = tan_bias * ty
        for k in range(TRAIL_STEPS):
            gx, gy = _grad_phi(x, y)
            vx -= gx * 2.5 * STEP
            vy -= gy * 2.5 * STEP
            sp = math.hypot(vx, vy)
            if sp > 5.0:
                vx *= 5.0 / sp
                vy *= 5.0 / sp
            nx = x + vx * STEP
            ny = y + vy * STEP
            # Trail intensity ramps along the trail: head bright, tail dim.
            tail_frac = (TRAIL_STEPS - k) / TRAIL_STEPS
            core_alpha = int(180 * tail_frac)
            halo_alpha = int(60 * tail_frac)
            # Cyan-white spectrum near bodies, blue-violet on outer rim.
            local_r = math.hypot(nx - BODY_A["x"], ny - BODY_A["y"])
            blend = min(1.0, 200.0 / (local_r + 50))
            r_col = int(140 + 100 * blend)
            g_col = int(200 + 50 * blend)
            b_col = 255
            cdraw.line([(x, y), (nx, ny)], fill=(r_col, g_col, b_col, core_alpha), width=1)
            hdraw.line([(x, y), (nx, ny)], fill=(r_col, g_col, b_col, halo_alpha), width=3)
            x, y = nx, ny
            d1 = math.hypot(x - BODY_A["x"], y - BODY_A["y"])
            d2 = math.hypot(x - BODY_B["x"], y - BODY_B["y"])
            if d1 < BODY_A["r"] or d2 < BODY_B["r"]:
                break
    # Apply blurs at different scales for proper bloom.
    halo = halo.filter(ImageFilter.GaussianBlur(4.0))
    core = core.filter(ImageFilter.GaussianBlur(0.6))
    im.alpha_composite(halo)
    im.alpha_composite(core)
    _bodies(im)
    _header(
        im,
        "Particle Flow",
        "gravitational streamlines · particles advect along force field · luminous ambiance",
    )
    path = os.path.join(OUT_DIR, "mockup_flow.png")
    im.save(path)
    return path


# ───────────────────────────────────────────────────────────────────
# MOCKUP 4 — Mixed: 2D grid warp + sparse particle flow (hybrid)
# ───────────────────────────────────────────────────────────────────
def mockup_mix():
    """2D grid as primary readout + ~600 sparse glowing particles
    drifting along force lines on top. Best of both worlds."""
    from PIL import ImageFilter
    im = Image.new("RGBA", (W, H), BG + (255,))
    draw = ImageDraw.Draw(im, "RGBA")
    # ── 2D grid pass (same as mockup_2d but slightly more transparent
    # so the particle layer reads on top) ─────────────────────────────
    cols, rows = 56, 34
    Y_TOP, Y_BOTTOM = 130, 760
    X_LEFT, X_RIGHT = 60, W - 60
    cw = (X_RIGHT - X_LEFT) / (cols - 1)
    rh = (Y_BOTTOM - Y_TOP) / (rows - 1)
    pts = [[(0.0, 0.0)] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            x = X_LEFT + c * cw
            y = Y_TOP + r * rh
            gx, gy = _grad_phi(x, y)
            scale = 14000.0
            dx = -gx * scale
            dy = -gy * scale
            mag = math.sqrt(dx * dx + dy * dy)
            cap = 28.0
            if mag > cap:
                dx *= cap / mag
                dy *= cap / mag
            pts[r][c] = (x + dx, y + dy)
    for r in range(rows):
        for c in range(cols - 1):
            (x1, y1), (x2, y2) = pts[r][c], pts[r][c + 1]
            draw.line([(x1, y1), (x2, y2)], fill=(140, 180, 230, 95), width=1)
    for c in range(cols):
        for r in range(rows - 1):
            (x1, y1), (x2, y2) = pts[r][c], pts[r + 1][c]
            draw.line([(x1, y1), (x2, y2)], fill=(140, 180, 230, 95), width=1)
    # ── Sparse particles (≈1/10th of mockup_flow density) ──
    halo = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    core = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    cdraw = ImageDraw.Draw(core, "RGBA")
    hdraw = ImageDraw.Draw(halo, "RGBA")
    rng = random.Random(11)
    NUM_PARTICLES = 600
    TRAIL_STEPS = 22
    STEP = 1.8
    for _ in range(NUM_PARTICLES):
        for _try in range(8):
            x = rng.uniform(20, W - 20)
            y = rng.uniform(115, H - 30)
            d1 = math.hypot(x - BODY_A["x"], y - BODY_A["y"])
            d2 = math.hypot(x - BODY_B["x"], y - BODY_B["y"])
            if d1 > BODY_A["r"] + 25 and d2 > BODY_B["r"] + 25:
                break
        nearest = BODY_A if d1 < d2 else BODY_B
        rx = x - nearest["x"]
        ry = y - nearest["y"]
        rmag = math.hypot(rx, ry) + 1e-6
        tx, ty = -ry / rmag, rx / rmag
        tan_bias = rng.uniform(-1.2, 1.2)
        vx = tan_bias * tx
        vy = tan_bias * ty
        for k in range(TRAIL_STEPS):
            gx, gy = _grad_phi(x, y)
            vx -= gx * 2.5 * STEP
            vy -= gy * 2.5 * STEP
            sp = math.hypot(vx, vy)
            if sp > 5.0:
                vx *= 5.0 / sp
                vy *= 5.0 / sp
            nx = x + vx * STEP
            ny = y + vy * STEP
            tail_frac = (TRAIL_STEPS - k) / TRAIL_STEPS
            core_alpha = int(200 * tail_frac)
            halo_alpha = int(45 * tail_frac)
            cdraw.line([(x, y), (nx, ny)], fill=(200, 230, 255, core_alpha), width=1)
            hdraw.line([(x, y), (nx, ny)], fill=(180, 220, 255, halo_alpha), width=3)
            x, y = nx, ny
            d1 = math.hypot(x - BODY_A["x"], y - BODY_A["y"])
            d2 = math.hypot(x - BODY_B["x"], y - BODY_B["y"])
            if d1 < BODY_A["r"] or d2 < BODY_B["r"]:
                break
    halo = halo.filter(ImageFilter.GaussianBlur(3.0))
    core = core.filter(ImageFilter.GaussianBlur(0.5))
    im.alpha_composite(halo)
    im.alpha_composite(core)
    _bodies(im)
    _header(
        im,
        "Hybrid: 2D Grid + Sparse Particles",
        "grid carries directional info · particles add motion + atmosphere",
    )
    path = os.path.join(OUT_DIR, "mockup_mix.png")
    im.save(path)
    return path


# ───────────────────────────────────────────────────────────────────
# MOCKUP 5 — Wrap-mode demo (toroidal continuation)
# ───────────────────────────────────────────────────────────────────
def mockup_wrap():
    """Show how 3D-rubber-sheet AND 2D-flat behave in wrap mode.
    The body near the LEFT edge has a 'ghost' copy at +W, so its
    well continues smoothly across the right edge. Same for top/bottom.
    Render side-by-side: top half = 3D rubber sheet wrap, bottom = 2D.
    """
    im = Image.new("RGBA", (W, H), BG + (255,))
    draw = ImageDraw.Draw(im, "RGBA")
    # Single body near the left edge for clarity.
    EDGE_BODY = {"x": 110, "y": 0, "r": 30, "m": 220, "color": (255, 150, 90)}
    # Wrap viewport size matches W,H for ghost replicas.
    GHOSTS = [(-W, 0), (0, 0), (W, 0)]   # 1D-horizontal only for clarity
    HALF_H = (H - 130) // 2
    BAND_TOP_Y = 135
    BAND_GAP = 30
    BAND2_TOP_Y = BAND_TOP_Y + HALF_H + BAND_GAP

    def _wrap_phi(x, y, body, body_y):
        p = 0.0
        for (ox, _oy) in GHOSTS:
            dx = x - (body["x"] + ox)
            dy = y - body_y
            r2 = dx * dx + dy * dy + 25 * 25
            r = math.sqrt(r2)
            falloff = math.exp(-r / 200.0)
            p += body["m"] * falloff / r
        return p

    def _wrap_grad(x, y, body, body_y):
        gx, gy = 0.0, 0.0
        for (ox, _oy) in GHOSTS:
            dx = x - (body["x"] + ox)
            dy = y - body_y
            r2 = dx * dx + dy * dy + 25 * 25
            invR3 = 1.0 / (r2 * math.sqrt(r2))
            gx += body["m"] * dx * invR3
            gy += body["m"] * dy * invR3
        return gx, gy

    def _draw_band(y_top, y_bot, mode_3d, label):
        cols, rows = 60, 22
        X_LEFT, X_RIGHT = 30, W - 30
        cw = (X_RIGHT - X_LEFT) / (cols - 1)
        rh = (y_bot - y_top) / (rows - 1)
        body_y = (y_top + y_bot) / 2
        pts = [[(0.0, 0.0)] * cols for _ in range(rows)]
        for r in range(rows):
            for c in range(cols):
                x = X_LEFT + c * cw
                y = y_top + r * rh
                if mode_3d:
                    depth = _wrap_phi(x, y, EDGE_BODY, body_y) * 60.0
                    pts[r][c] = (x, y + depth)
                else:
                    gx, gy = _wrap_grad(x, y, EDGE_BODY, body_y)
                    scale = 14000.0
                    dx = -gx * scale
                    dy = -gy * scale
                    mag = math.sqrt(dx * dx + dy * dy)
                    cap = 28.0
                    if mag > cap:
                        dx *= cap / mag
                        dy *= cap / mag
                    pts[r][c] = (x + dx, y + dy)
        for r in range(rows):
            for c in range(cols - 1):
                (x1, y1), (x2, y2) = pts[r][c], pts[r][c + 1]
                draw.line([(x1, y1), (x2, y2)], fill=(150, 195, 240, 150), width=1)
        for c in range(cols):
            for r in range(rows - 1):
                (x1, y1), (x2, y2) = pts[r][c], pts[r + 1][c]
                draw.line([(x1, y1), (x2, y2)], fill=(150, 195, 240, 150), width=1)
        # Draw body
        b = EDGE_BODY
        for i in range(6, 0, -1):
            rr = b["r"] + i * 4
            draw.ellipse(
                [b["x"] - rr, body_y - rr, b["x"] + rr, body_y + rr],
                fill=(*b["color"], max(0, 30 - i * 4)),
            )
        draw.ellipse(
            [b["x"] - b["r"], body_y - b["r"], b["x"] + b["r"], body_y + b["r"]],
            fill=(*b["color"], 255),
            outline=(255, 255, 255, 90), width=1,
        )
        # Visualize wrap boundary as dashed vertical at far right.
        for yy in range(int(y_top), int(y_bot), 8):
            draw.line([(W - 30, yy), (W - 30, yy + 4)], fill=(200, 100, 100, 180), width=1)
            draw.line([(30, yy), (30, yy + 4)], fill=(200, 100, 100, 180), width=1)
        # Band label
        draw.text((40, y_top - 22), label, font=FONT_CAPTION, fill=(200, 220, 240, 240))

    _draw_band(BAND_TOP_Y, BAND_TOP_Y + HALF_H, mode_3d=True,
               label="3D — well continues across the wrap boundary (red dashed)")
    _draw_band(BAND2_TOP_Y, BAND2_TOP_Y + HALF_H, mode_3d=False,
               label="2D — grid pull continues across the wrap boundary")
    _header(
        im,
        "Wrap Mode",
        "ghost replicas at ±W make the field continuous across the toroidal boundary",
    )
    path = os.path.join(OUT_DIR, "mockup_wrap.png")
    im.save(path)
    return path


# ───────────────────────────────────────────────────────────────────
# MOCKUP 6 — Hard cap vs smooth saturation at m=1000 (bug fix demo)
# ───────────────────────────────────────────────────────────────────
def mockup_hard_vs_smooth():
    """Side-by-side demonstration of the high-mass bug.

    Top band:    hard cap (current bug)  — vertices all saturate to 28 px,
                 grid folds + no differential.
    Bottom band: smooth rational saturation (fix) — preserves the
                 gradient at all magnitudes → clean compression.
    Single mass=1000 body at the center of each band.
    """
    im = Image.new("RGBA", (W, H), BG + (255,))
    draw = ImageDraw.Draw(im, "RGBA")
    HEAVY_BODY = {"x": 300, "y": 0, "r": 32, "m": 1000, "color": (255, 130, 70)}

    def _heavy_grad(x, y, body_y):
        ez = 80.0 * 1000.0
        dx = x - HEAVY_BODY["x"]
        dy = y - body_y
        r2 = dx * dx + dy * dy + 16.0
        invR3 = 1.0 / (r2 * math.sqrt(r2))
        return (-ez * dx * invR3, -ez * dy * invR3)

    def _draw_band(y_top, y_bot, mode_smooth, label):
        cols, rows = 56, 28
        X_LEFT, X_RIGHT = 30, 600
        cw = (X_RIGHT - X_LEFT) / (cols - 1)
        rh = (y_bot - y_top) / (rows - 1)
        body_y = (y_top + y_bot) / 2
        pts = [[(0.0, 0.0)] * cols for _ in range(rows)]
        DISP_SCALE = 175.0
        CAP = 35.0 if mode_smooth else 28.0
        for r in range(rows):
            for c in range(cols):
                x = X_LEFT + c * cw
                y = y_top + r * rh
                gx, gy = _heavy_grad(x, y, body_y)
                disp_x = gx * DISP_SCALE
                disp_y = gy * DISP_SCALE
                mag = math.sqrt(disp_x * disp_x + disp_y * disp_y)
                if mode_smooth:
                    factor = CAP / (CAP + mag)
                else:
                    factor = (CAP / mag) if mag > CAP else 1.0
                pts[r][c] = (x + disp_x * factor, y + disp_y * factor)
        for r in range(rows):
            for c in range(cols - 1):
                (x1, y1), (x2, y2) = pts[r][c], pts[r][c + 1]
                draw.line([(x1, y1), (x2, y2)], fill=(150, 195, 240, 150), width=1)
        for c in range(cols):
            for r in range(rows - 1):
                (x1, y1), (x2, y2) = pts[r][c], pts[r + 1][c]
                draw.line([(x1, y1), (x2, y2)], fill=(150, 195, 240, 150), width=1)
        b = HEAVY_BODY
        for i in range(6, 0, -1):
            rr = b["r"] + i * 4
            draw.ellipse(
                [b["x"] - rr, body_y - rr, b["x"] + rr, body_y + rr],
                fill=(*b["color"], max(0, 30 - i * 4)),
            )
        draw.ellipse(
            [b["x"] - b["r"], body_y - b["r"], b["x"] + b["r"], body_y + b["r"]],
            fill=(*b["color"], 255),
            outline=(255, 255, 255, 90), width=1,
        )
        draw.text((X_LEFT, y_top - 22), label, font=FONT_CAPTION, fill=(220, 230, 240, 240))

    HALF_H = (H - 130 - 40) // 2
    _draw_band(140, 140 + HALF_H, mode_smooth=False,
               label="HARD CAP (bug): vertices saturate → grid folds + no differential")
    _draw_band(140 + HALF_H + 40, 140 + 2 * HALF_H + 40, mode_smooth=True,
               label="SMOOTH SATURATION (fix): displacement preserves gradient → grid compresses cleanly")
    _header(
        im,
        "Bug Fix — m=1000 body",
        "hard-cap vs rational saturation · same gravity field, same scale",
    )
    path = os.path.join(OUT_DIR, "mockup_bugfix.png")
    im.save(path)
    return path


if __name__ == "__main__":
    p1 = mockup_3d()
    p2 = mockup_2d()
    p3 = mockup_flow()
    p4 = mockup_mix()
    p5 = mockup_wrap()
    p6 = mockup_hard_vs_smooth()
    print("Wrote:")
    for p in (p1, p2, p3, p4, p5, p6):
        print(" ", p)
