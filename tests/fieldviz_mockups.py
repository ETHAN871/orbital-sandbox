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

    def _draw_band(y_top, y_bot, mode_label, label):
        """mode_label ∈ {'hard', 'smooth', 'antiovershoot', 'atan'}.

        Single-body bands show the saturation function shape.
        See mockup_cluster_depth for the multi-body cluster comparison
        that motivated the atan switch.
        """
        cols, rows = 56, 28
        X_LEFT, X_RIGHT = 30, 600
        cw = (X_RIGHT - X_LEFT) / (cols - 1)
        rh = (y_bot - y_top) / (rows - 1)
        body_y = (y_top + y_bot) / 2
        pts = [[(0.0, 0.0)] * cols for _ in range(rows)]
        DISP_SCALE = 175.0
        CAP = 35.0
        OVERSHOOT_FRAC = 0.4
        L_ATAN = 50.0
        for r in range(rows):
            for c in range(cols):
                x = X_LEFT + c * cw
                y = y_top + r * rh
                dx = x - HEAVY_BODY["x"]
                dy = y - body_y
                r2 = dx * dx + dy * dy + 16.0
                r_dist = math.sqrt(r2)
                ez = 80.0 * 1000.0
                if mode_label == 'hard':
                    grad_x = -ez * dx / (r2 * r_dist)
                    grad_y = -ez * dy / (r2 * r_dist)
                    disp_x = grad_x * DISP_SCALE
                    disp_y = grad_y * DISP_SCALE
                    mag = math.sqrt(disp_x * disp_x + disp_y * disp_y)
                    factor = (28.0 / mag) if mag > 28.0 else 1.0
                    pts[r][c] = (x + disp_x * factor, y + disp_y * factor)
                elif mode_label == 'smooth':
                    grad_x = -ez * dx / (r2 * r_dist)
                    grad_y = -ez * dy / (r2 * r_dist)
                    disp_x = grad_x * DISP_SCALE
                    disp_y = grad_y * DISP_SCALE
                    mag = math.sqrt(disp_x * disp_x + disp_y * disp_y)
                    factor = CAP / (CAP + mag)
                    pts[r][c] = (x + disp_x * factor, y + disp_y * factor)
                elif mode_label == 'antiovershoot':
                    raw_mag = ez * DISP_SCALE / r2
                    max_allowed = r_dist * OVERSHOOT_FRAC
                    bounded = min(raw_mag, max_allowed)
                    ux, uy = -dx / r_dist, -dy / r_dist
                    disp_x = ux * bounded
                    disp_y = uy * bounded
                    mag = math.sqrt(disp_x * disp_x + disp_y * disp_y)
                    factor = CAP / (CAP + mag)
                    pts[r][c] = (x + disp_x * factor, y + disp_y * factor)
                else:  # 'atan' — anti-overshoot + atan saturation
                    raw_mag = ez * DISP_SCALE / r2
                    max_allowed = r_dist * OVERSHOOT_FRAC
                    bounded = min(raw_mag, max_allowed)
                    ux, uy = -dx / r_dist, -dy / r_dist
                    disp_x = ux * bounded
                    disp_y = uy * bounded
                    mag = math.sqrt(disp_x * disp_x + disp_y * disp_y)
                    sat = L_ATAN * math.atan(mag / L_ATAN) if mag > 1e-6 else 0.0
                    factor = sat / mag if mag > 1e-4 else 1.0
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

    GAP = 30
    y0 = 140
    THIRD_H = (H - y0 - 30 - 2 * GAP) // 3
    _draw_band(
        y0, y0 + THIRD_H, mode_label='hard',
        label="HARD CAP (old): vertices clamp at 28 px → grid folds + no differential near body",
    )
    _draw_band(
        y0 + THIRD_H + GAP, y0 + 2 * THIRD_H + GAP, mode_label='smooth',
        label="SMOOTH SATURATION (prior fix): no fold far out, but vertices still cross body → triangles",
    )
    _draw_band(
        y0 + 2 * (THIRD_H + GAP), y0 + 3 * THIRD_H + 2 * GAP,
        mode_label='antiovershoot',
        label="ANTI-OVERSHOOT (new): per-body bound = 0.4·r → no body crossing, no triangles, no folds",
    )
    _header(
        im,
        "Bug Fix — m=1000 body (3-way: hard → smooth → anti-overshoot)",
        "shows triangles + folding root cause is per-body overshoot, not global cap shape",
    )
    path = os.path.join(OUT_DIR, "mockup_bugfix.png")
    im.save(path)
    return path


# ───────────────────────────────────────────────────────────────────
# MOCKUP 7 — Cluster depth: rational vs atan saturation
# ───────────────────────────────────────────────────────────────────
def mockup_cluster_depth():
    """Side-by-side comparison of cap/(cap+m) vs L·atan(m/L) when
    many bodies are clustered (the user-reported scenario).

    Top band: rational sat cap/(cap+mag), cap=35 → cluster warps no
              deeper than a single body. Flat-looking despite mass.
    Bottom band: atan sat L·atan(mag/L), L=50 (asymptote ≈ 78 px) →
              cluster produces a visibly deeper well.

    Same 90-body diagonal-chain cluster mirroring user's screenshot.
    """
    im = Image.new("RGBA", (W, H), BG + (255,))
    draw = ImageDraw.Draw(im, "RGBA")

    # Build a ~90-body diagonal-chain cluster (mimics user screenshot).
    cluster = []
    rng = random.Random(7)
    for row in range(15):
        for col in range(6):
            # Diagonal offset so it forms a chain.
            cx = 220 + row * 14 + col * 14 + rng.uniform(-2, 2)
            cy = 90 + row * 22 + col * 4 + rng.uniform(-2, 2)
            cluster.append({"x": cx, "y": cy, "r": 10, "m": 50})

    def _cluster_grad(x, y, body_y_offset):
        ez_per = 80.0 * 50.0
        gx, gy = 0.0, 0.0
        for b in cluster:
            dx = x - b["x"]
            dy = y - (b["y"] + body_y_offset)
            r2 = dx * dx + dy * dy + 16.0
            r_dist = math.sqrt(r2)
            raw_mag = ez_per * 175.0 / r2
            max_allowed = r_dist * 0.4
            bounded = min(raw_mag, max_allowed)
            gx += -dx / r_dist * bounded
            gy += -dy / r_dist * bounded
        return gx, gy

    def _draw_band(y_top, y_bot, sat_label, label):
        cols, rows = 70, 36
        X_LEFT, X_RIGHT = 30, W - 30
        cw = (X_RIGHT - X_LEFT) / (cols - 1)
        rh = (y_bot - y_top) / (rows - 1)
        # Offset cluster so it sits inside the band.
        body_y_offset = y_top - 90
        pts = [[(0.0, 0.0)] * cols for _ in range(rows)]
        CAP = 35.0
        L = 50.0
        for r in range(rows):
            for c in range(cols):
                x = X_LEFT + c * cw
                y = y_top + r * rh
                disp_x, disp_y = _cluster_grad(x, y, body_y_offset)
                mag = math.sqrt(disp_x * disp_x + disp_y * disp_y)
                if sat_label == 'rational':
                    factor = CAP / (CAP + mag)
                else:  # atan
                    sat = L * math.atan(mag / L) if mag > 1e-6 else 0.0
                    factor = sat / mag if mag > 1e-4 else 1.0
                pts[r][c] = (x + disp_x * factor, y + disp_y * factor)
        for r in range(rows):
            for c in range(cols - 1):
                (x1, y1), (x2, y2) = pts[r][c], pts[r][c + 1]
                draw.line([(x1, y1), (x2, y2)], fill=(150, 195, 240, 150), width=1)
        for c in range(cols):
            for r in range(rows - 1):
                (x1, y1), (x2, y2) = pts[r][c], pts[r + 1][c]
                draw.line([(x1, y1), (x2, y2)], fill=(150, 195, 240, 150), width=1)
        # Draw the cluster bodies.
        body_colors = [
            (255, 100, 150), (100, 200, 255), (200, 100, 255),
            (100, 255, 150), (255, 180, 80), (150, 100, 255),
            (255, 220, 80), (80, 255, 200),
        ]
        for i, b in enumerate(cluster):
            col = body_colors[i % len(body_colors)]
            cx, cy = b["x"], b["y"] + body_y_offset
            draw.ellipse(
                [cx - b["r"], cy - b["r"], cx + b["r"], cy + b["r"]],
                fill=(*col, 255),
                outline=(255, 255, 255, 90), width=1,
            )
        draw.text((30, y_top - 24), label, font=FONT_CAPTION, fill=(220, 230, 240, 240))

    HALF_H = (H - 130 - 40) // 2
    _draw_band(140, 140 + HALF_H, sat_label='rational',
               label="RATIONAL cap/(cap+m), cap=35 — cluster cap-bound, well stays shallow (current bug)")
    _draw_band(140 + HALF_H + 40, 140 + 2 * HALF_H + 40, sat_label='atan',
               label="ATAN L·atan(m/L), L=50 (asymp ≈ 78 px) — cluster carves visibly deeper well (fix)")
    _header(
        im,
        "Cluster depth — rational vs atan saturation (90-body chain)",
        "user-reported: many bodies stacked → mild warp. atan asymptote ≈ 2× rational cap.",
    )
    path = os.path.join(OUT_DIR, "mockup_cluster_depth.png")
    im.save(path)
    return path


# ───────────────────────────────────────────────────────────────────
# Shared shape helpers for marching-squares (used by both option mockups).
# ───────────────────────────────────────────────────────────────────
def _marching_squares(grid, rows, cols, GS, thresholds, draw, color):
    for thr in thresholds:
        for r in range(rows - 1):
            for c in range(cols - 1):
                v00 = grid[r][c]
                v10 = grid[r][c + 1]
                v01 = grid[r + 1][c]
                v11 = grid[r + 1][c + 1]
                case = 0
                if v00 > thr: case |= 1
                if v10 > thr: case |= 2
                if v11 > thr: case |= 4
                if v01 > thr: case |= 8
                if case == 0 or case == 15:
                    continue
                def lerp(va, vb, xa, ya, xb, yb):
                    t = 0.5 if vb == va else max(0, min(1, (thr - va) / (vb - va)))
                    return (xa + (xb - xa) * t, ya + (yb - ya) * t)
                x0, y0 = c * GS, r * GS
                p_top = lerp(v00, v10, x0, y0, x0 + GS, y0)
                p_right = lerp(v10, v11, x0 + GS, y0, x0 + GS, y0 + GS)
                p_bot = lerp(v01, v11, x0, y0 + GS, x0 + GS, y0 + GS)
                p_left = lerp(v00, v01, x0, y0, x0, y0 + GS)
                segs = []
                if case in (1, 14): segs = [(p_top, p_left)]
                elif case in (2, 13): segs = [(p_top, p_right)]
                elif case in (4, 11): segs = [(p_right, p_bot)]
                elif case in (8, 7):  segs = [(p_left, p_bot)]
                elif case in (3, 12): segs = [(p_left, p_right)]
                elif case in (6, 9):  segs = [(p_top, p_bot)]
                elif case == 5: segs = [(p_top, p_right), (p_left, p_bot)]
                elif case == 10: segs = [(p_top, p_left), (p_right, p_bot)]
                for (s, e) in segs:
                    draw.line([s, e], fill=color, width=1)


# ───────────────────────────────────────────────────────────────────
# MOCKUP 8 — Option 3 reference: Jobard-Lefer evenly-spaced streamlines
# ───────────────────────────────────────────────────────────────────
def _grad_phi_multi(x, y, bodies, eps2):
    gx, gy = 0.0, 0.0
    for b in bodies:
        dx = b["x"] - x
        dy = b["y"] - y
        r2 = dx * dx + dy * dy + eps2
        r = math.sqrt(r2)
        k = b["m"] / (r2 * r)
        gx += k * dx
        gy += k * dy
    return gx, gy


def _trace_streamline(sx, sy, bodies, eps2, step, max_steps, vw, vh, radii):
    def step_dir(px, py, sign):
        gx, gy = _grad_phi_multi(px, py, bodies, eps2)
        gmag = math.hypot(gx, gy)
        if gmag < 1e-6: return None
        return (sign * gx / gmag, sign * gy / gmag)
    poly = [(sx, sy)]
    for sign, prepend in ((1, False), (-1, True)):
        px, py = sx, sy
        local = []
        for _ in range(max_steps):
            d = step_dir(px, py, sign)
            if d is None: break
            mx, my = px + d[0] * step * 0.5, py + d[1] * step * 0.5
            d2 = step_dir(mx, my, sign)
            if d2 is None: break
            nx, ny = px + d2[0] * step, py + d2[1] * step
            if nx < 0 or nx > vw or ny < 0 or ny > vh:
                local.append((nx, ny))
                break
            hit = False
            for b, r in zip(bodies, radii):
                if (nx - b["x"]) ** 2 + (ny - b["y"]) ** 2 < r * r:
                    hit = True; break
            local.append((nx, ny))
            if hit: break
            px, py = nx, ny
        if prepend:
            poly = list(reversed(local)) + poly
        else:
            poly = poly + local
    return poly


def _too_close(x, y, occ, cell, d_sep):
    cx = int(x / cell)
    cy = int(y / cell)
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            pts = occ.get((cx + ox, cy + oy))
            if pts is None: continue
            for (px, py) in pts:
                if (px - x) ** 2 + (py - y) ** 2 < d_sep * d_sep:
                    return True
    return False


# Shared scene for option mockups so they compare apples-to-apples.
_OPT_BODIES = [
    {"x": 320, "y": 300, "r": 22, "m": 200, "color": (255, 120, 100)},
    {"x": 680, "y": 380, "r": 30, "m": 350, "color": (140, 180, 255)},
    {"x": 520, "y": 600, "r": 18, "m": 120, "color": (180, 240, 140)},
]
_OPT_EPS2 = 16.0


def _draw_bodies(draw, bodies):
    for b in bodies:
        for i in range(6, 0, -1):
            rr = b["r"] + i * 4
            draw.ellipse(
                [b["x"] - rr, b["y"] - rr, b["x"] + rr, b["y"] + rr],
                fill=(*b["color"], max(0, 30 - i * 4)),
            )
        draw.ellipse(
            [b["x"] - b["r"], b["y"] - b["r"], b["x"] + b["r"], b["y"] + b["r"]],
            fill=(*b["color"], 255),
            outline=(255, 255, 255, 140), width=2,
        )


def mockup_option3():
    im = Image.new("RGBA", (W, H), BG + (255,))
    draw = ImageDraw.Draw(im, "RGBA")
    bodies = _OPT_BODIES
    eps2 = _OPT_EPS2
    radii = [b["r"] for b in bodies]

    def phi_at(x, y):
        p = 0.0
        for b in bodies:
            dx = x - b["x"]; dy = y - b["y"]
            p += -b["m"] / math.sqrt(dx * dx + dy * dy + eps2)
        return p
    phi_max = max(abs(phi_at(b["x"], b["y"])) for b in bodies)

    GS = 4
    cols = W // GS
    rows = H // GS
    phi_grid = [[-phi_at(c * GS, r * GS) for c in range(cols)] for r in range(rows)]
    NUM_RINGS = 9
    K = (phi_max / 0.4) ** (1 / (NUM_RINGS - 1))
    phi_thr = [0.4 * (K ** n) for n in range(NUM_RINGS)]
    _marching_squares(phi_grid, rows, cols, GS, phi_thr, draw, (170, 210, 250, 110))

    # Jobard-Lefer streamline placement.
    D_SEP = 28.0
    STEP = 4
    MAX_STEPS = 200
    cell = D_SEP
    occ = {}

    def push_occ(poly):
        for (x, y) in poly:
            occ.setdefault((int(x / cell), int(y / cell)), []).append((x, y))

    streamlines = []
    sl0 = _trace_streamline(W // 2 + 40, H // 2 + 30, bodies, eps2, STEP, MAX_STEPS, W, H, radii)
    streamlines.append(sl0)
    push_occ(sl0)

    queue = list(sl0)
    while queue and len(streamlines) < 600:
        sx, sy = queue.pop(0)
        gx, gy = _grad_phi_multi(sx, sy, bodies, eps2)
        gmag = math.hypot(gx, gy)
        if gmag < 1e-6: continue
        nxd, nyd = -gy / gmag, gx / gmag
        for sign in (-1, 1):
            cx = sx + nxd * D_SEP * sign
            cy = sy + nyd * D_SEP * sign
            if cx < 0 or cx > W or cy < 0 or cy > H: continue
            inside = False
            for b, r in zip(bodies, radii):
                if (cx - b["x"]) ** 2 + (cy - b["y"]) ** 2 < (r * 1.5) ** 2:
                    inside = True; break
            if inside: continue
            if _too_close(cx, cy, occ, cell, D_SEP * 0.85): continue
            sl = _trace_streamline(cx, cy, bodies, eps2, STEP, MAX_STEPS, W, H, radii)
            if len(sl) < 4: continue
            streamlines.append(sl)
            push_occ(sl)
            for k in range(0, len(sl), 6):
                queue.append(sl[k])

    for sl in streamlines:
        if len(sl) < 2: continue
        pts = [(int(p[0]), int(p[1])) for p in sl]
        draw.line(pts, fill=(195, 225, 255, 200), width=1, joint='curve')

    _draw_bodies(draw, bodies)
    _header(
        im,
        "Option 3 — Jobard-Lefer evenly-spaced streamlines + equipotential",
        "field lines spaced by placement algorithm (no radial clustering) + log-spaced contour rings",
    )
    path = os.path.join(OUT_DIR, "mockup_option3.png")
    im.save(path)
    return path


# ───────────────────────────────────────────────────────────────────
# MOCKUP 9 — Option 4: marching-squares dual scalar fields
# ───────────────────────────────────────────────────────────────────
def mockup_option4():
    im = Image.new("RGBA", (W, H), BG + (255,))
    draw = ImageDraw.Draw(im, "RGBA")
    bodies = _OPT_BODIES
    eps2 = _OPT_EPS2

    def phi_at(x, y):
        p = 0.0
        for b in bodies:
            dx = x - b["x"]; dy = y - b["y"]
            p += -b["m"] / math.sqrt(dx * dx + dy * dy + eps2)
        return p

    def psi_at(x, y):
        s = 0.0
        for b in bodies:
            dx = x - b["x"]; dy = y - b["y"]
            s += b["m"] * math.atan2(dy, dx)
        return s

    phi_max = max(abs(phi_at(b["x"], b["y"])) for b in bodies)
    GS = 4
    cols = W // GS
    rows = H // GS
    phi_grid = [[-phi_at(c * GS, r * GS) for c in range(cols)] for r in range(rows)]
    psi_grid = [[psi_at(c * GS, r * GS)  for c in range(cols)] for r in range(rows)]

    NUM_RINGS = 11
    K = (phi_max / 0.35) ** (1 / (NUM_RINGS - 1))
    phi_thr = [0.35 * (K ** n) for n in range(NUM_RINGS)]
    _marching_squares(phi_grid, rows, cols, GS, phi_thr, draw, (170, 210, 250, 140))

    psi_min = min(min(row) for row in psi_grid)
    psi_max = max(max(row) for row in psi_grid)
    NUM_PSI = 22
    psi_thr = [psi_min + (psi_max - psi_min) * (i + 0.5) / NUM_PSI for i in range(NUM_PSI)]
    _marching_squares(psi_grid, rows, cols, GS, psi_thr, draw, (255, 195, 165, 110))

    _draw_bodies(draw, bodies)
    _header(
        im,
        "Option 4 — marching-squares dual: φ (cyan) + approximate ψ (orange)",
        "two scalar fields' level sets compose orthogonal curvilinear grid; ψ has small atan2 branch artifacts",
    )
    path = os.path.join(OUT_DIR, "mockup_option4.png")
    im.save(path)
    return path


# ───────────────────────────────────────────────────────────────────
# Re-research mockups: LIC, hypsometric tint, hillshade
# (techniques that avoid the line-topology issues we keep hitting)
# ───────────────────────────────────────────────────────────────────

def _build_field_grids(bodies, eps2, GS):
    """Sample φ, F_x, F_y, |F| on a coarse grid for use by mockups.
    Uses numpy if available for speed; falls back to pure-Python loops.
    Returns (phi, fx, fy, fmag) as 2D lists or numpy arrays."""
    try:
        import numpy as np
        cols = W // GS + 1
        rows = H // GS + 1
        xs = (np.arange(cols) * GS).reshape(1, cols)
        ys = (np.arange(rows) * GS).reshape(rows, 1)
        phi = np.zeros((rows, cols), dtype=np.float64)
        fx  = np.zeros((rows, cols), dtype=np.float64)
        fy  = np.zeros((rows, cols), dtype=np.float64)
        for b in bodies:
            dx = b["x"] - xs   # (rows, cols)
            dy = b["y"] - ys
            r2 = dx * dx + dy * dy + eps2
            r = np.sqrt(r2)
            phi += -b["m"] / r
            k = b["m"] / (r2 * r)
            fx += k * dx
            fy += k * dy
        fmag = np.sqrt(fx * fx + fy * fy)
        return phi, fx, fy, fmag, True
    except ImportError:
        cols = W // GS + 1
        rows = H // GS + 1
        phi  = [[0.0] * cols for _ in range(rows)]
        fx   = [[0.0] * cols for _ in range(rows)]
        fy   = [[0.0] * cols for _ in range(rows)]
        fmag = [[0.0] * cols for _ in range(rows)]
        for j in range(rows):
            yy = j * GS
            for i in range(cols):
                xx = i * GS
                p = 0.0; ax = 0.0; ay = 0.0
                for b in bodies:
                    dx = b["x"] - xx
                    dy = b["y"] - yy
                    r2 = dx * dx + dy * dy + eps2
                    r  = math.sqrt(r2)
                    p += -b["m"] / r
                    k = b["m"] / (r2 * r)
                    ax += k * dx
                    ay += k * dy
                phi[j][i]  = p
                fx[j][i]   = ax
                fy[j][i]   = ay
                fmag[j][i] = math.sqrt(ax * ax + ay * ay)
        return phi, fx, fy, fmag, False


def _equipotential_overlay(draw, bodies, eps2, color=(170, 210, 250, 130)):
    """Render log-spaced equipotential rings via marching squares on
    a coarse grid. Used by all three re-research mockups."""
    GS = 4
    cols = W // GS
    rows = H // GS
    grid = [[0.0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            x, y = c * GS, r * GS
            p = 0.0
            for b in bodies:
                dx = x - b["x"]; dy = y - b["y"]
                p += -b["m"] / math.sqrt(dx * dx + dy * dy + eps2)
            grid[r][c] = -p
    phi_max = 0.0
    for bb in bodies:
        p = 0.0
        for b in bodies:
            dx = bb["x"] - b["x"]; dy = bb["y"] - b["y"]
            p += -b["m"] / math.sqrt(dx * dx + dy * dy + eps2)
        phi_max = max(phi_max, abs(p))
    NUM_RINGS = 10
    K = (phi_max / 0.4) ** (1 / (NUM_RINGS - 1))
    thr_list = [0.4 * (K ** n) for n in range(NUM_RINGS)]
    _marching_squares(grid, rows, cols, GS, thr_list, draw, color)


def mockup_lic():
    """Line Integral Convolution: a noise texture convolved along the
    field's streamlines. Produces a flowing 'brush stroke' pattern that
    visually traces direction without ever drawing discrete lines.
    Combined with equipotential rings."""
    try:
        import numpy as np
    except ImportError:
        print("[lic] numpy required for LIC mockup, skipping.")
        return None
    im = Image.new("RGBA", (W, H), BG + (255,))
    draw = ImageDraw.Draw(im, "RGBA")
    bodies = _OPT_BODIES
    eps2 = _OPT_EPS2

    # Low-res LIC: render at half-res then upsample (LIC is expensive).
    LIC_DOWN = 2
    Wl, Hl = W // LIC_DOWN, H // LIC_DOWN

    # Per-pixel field direction on the low-res grid.
    xs = (np.arange(Wl).reshape(1, Wl) + 0.5) * LIC_DOWN
    ys = (np.arange(Hl).reshape(Hl, 1) + 0.5) * LIC_DOWN
    fx = np.zeros((Hl, Wl), dtype=np.float64)
    fy = np.zeros((Hl, Wl), dtype=np.float64)
    for b in bodies:
        dx = b["x"] - xs
        dy = b["y"] - ys
        r2 = dx * dx + dy * dy + eps2
        r = np.sqrt(r2)
        k = b["m"] / (r2 * r)
        fx += k * dx
        fy += k * dy
    fmag = np.sqrt(fx * fx + fy * fy) + 1e-9
    ux = fx / fmag
    uy = fy / fmag

    # White noise texture (low-res).
    rng = np.random.default_rng(seed=1729)
    noise = rng.random((Hl, Wl)).astype(np.float64)

    # LIC convolution: walk forward + backward along field, average noise.
    L_STEPS = 18
    STEP_PX_LR = 1.0   # in low-res pixel units
    out = noise.copy()
    weight = 1.0
    # Pre-compute integer index grids.
    ii = np.arange(Wl).reshape(1, Wl)
    jj = np.arange(Hl).reshape(Hl, 1)
    for sign in (+1, -1):
        px = ii.astype(np.float64).copy() + np.zeros((Hl, Wl))
        py = jj.astype(np.float64).copy() + np.zeros((Hl, Wl))
        for _ in range(L_STEPS):
            # Advance along field direction (sampled at current px, py).
            xi = np.clip(px.astype(np.int32), 0, Wl - 1)
            yi = np.clip(py.astype(np.int32), 0, Hl - 1)
            dx = sign * ux[yi, xi] * STEP_PX_LR
            dy = sign * uy[yi, xi] * STEP_PX_LR
            px = px + dx
            py = py + dy
            xi = np.clip(px.astype(np.int32), 0, Wl - 1)
            yi = np.clip(py.astype(np.int32), 0, Hl - 1)
            out = out + noise[yi, xi]
            weight += 1.0
    out = out / weight
    # Boost contrast: histogram stretch.
    lo, hi = np.percentile(out, [5, 95])
    out = np.clip((out - lo) / max(hi - lo, 1e-6), 0, 1)
    # Map to grayscale image.
    img_lr = (out * 255).astype(np.uint8)
    # Upsample.
    from PIL import Image as PImg
    lic_img = PImg.fromarray(img_lr, mode='L').resize((W, H), PImg.BILINEAR)
    # Blend onto canvas: LIC as a faint flowing texture.
    # Use cyan tint, modulated by LIC brightness.
    lic_rgba = PImg.new("RGBA", (W, H))
    lic_pixels = lic_rgba.load()
    lic_grey = lic_img.load()
    for y in range(H):
        for x in range(W):
            v = lic_grey[x, y]
            # Cool cyan-blue tint, alpha modulated by brightness.
            lic_pixels[x, y] = (140 + v // 4, 180 + v // 4, 230, int(v * 0.6))
    im.alpha_composite(lic_rgba)

    _equipotential_overlay(draw, bodies, eps2, color=(220, 240, 255, 160))
    _draw_bodies(draw, bodies)
    _header(
        im,
        "LIC — Line Integral Convolution + equipotential rings",
        "noise convolved along field streamlines → flowing 'brush stroke' texture, no discrete lines",
    )
    path = os.path.join(OUT_DIR, "mockup_lic.png")
    im.save(path)
    return path


def mockup_hypsometric():
    """Hypsometric tint: color the potential field directly (deep φ =
    dark, shallow φ = light) as a smooth gradient. Plus equipotential
    rings on top."""
    try:
        import numpy as np
    except ImportError:
        print("[hypso] numpy required, skipping.")
        return None
    im = Image.new("RGBA", (W, H), BG + (255,))
    draw = ImageDraw.Draw(im, "RGBA")
    bodies = _OPT_BODIES
    eps2 = _OPT_EPS2

    xs = np.arange(W).reshape(1, W)
    ys = np.arange(H).reshape(H, 1)
    phi = np.zeros((H, W), dtype=np.float64)
    for b in bodies:
        dx = xs - b["x"]
        dy = ys - b["y"]
        r = np.sqrt(dx * dx + dy * dy + eps2)
        phi += -b["m"] / r
    # Map φ (negative, deep at body) to elevation-like color.
    # Deepest = darkest (deep blue); shallowest (φ→0) = bright cyan-white.
    depth = -phi   # positive, large near body
    lo, hi = np.percentile(depth, [5, 99])
    t = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    # Color ramp: dark navy → indigo → cool cyan-blue → near-white at peaks
    # (treat "deep gravity well" as the dark valley).
    r_arr = (10  + t * (40  - 10))   # dark
    g_arr = (15  + t * (60  - 15))
    b_arr = (25  + t * (150 - 25))
    # Actually invert: deep well = dark, far = bright (depth increases toward body)
    # So invert t for color:
    ti = 1 - t   # high = far from body, low = at body
    r_arr = (10  + ti * (130 - 10))
    g_arr = (12  + ti * (170 - 12))
    b_arr = (20  + ti * (230 - 20))
    rgb = np.stack([r_arr, g_arr, b_arr], axis=-1).astype(np.uint8)
    from PIL import Image as PImg
    tint_img = PImg.fromarray(rgb, mode='RGB').convert("RGBA")
    im.alpha_composite(tint_img)

    _equipotential_overlay(draw, bodies, eps2, color=(255, 255, 255, 90))
    _draw_bodies(draw, bodies)
    _header(
        im,
        "Hypsometric tint + equipotential rings",
        "φ as smooth color gradient (dark valley = deep well, light = flat) + log-spaced rings",
    )
    path = os.path.join(OUT_DIR, "mockup_hypsometric.png")
    im.save(path)
    return path


def mockup_hillshade():
    """Shaded relief: compute the 'surface normal' of the potential
    treated as a height map, dot with a light direction. Gives a 3D-lit
    topographic feel on a 2D plane. Plus equipotential rings."""
    try:
        import numpy as np
    except ImportError:
        print("[hillshade] numpy required, skipping.")
        return None
    im = Image.new("RGBA", (W, H), BG + (255,))
    draw = ImageDraw.Draw(im, "RGBA")
    bodies = _OPT_BODIES
    eps2 = _OPT_EPS2

    xs = np.arange(W).reshape(1, W)
    ys = np.arange(H).reshape(H, 1)
    phi = np.zeros((H, W), dtype=np.float64)
    for b in bodies:
        dx = xs - b["x"]
        dy = ys - b["y"]
        r = np.sqrt(dx * dx + dy * dy + eps2)
        phi += -b["m"] / r
    # "Height" = -phi (positive at body, 0 far away). Use it as terrain.
    height = -phi
    # Compute gradient (slope) via finite differences.
    dzdx = np.zeros_like(height)
    dzdy = np.zeros_like(height)
    dzdx[:, 1:-1] = (height[:, 2:] - height[:, :-2]) * 0.5
    dzdy[1:-1, :] = (height[2:, :] - height[:-2, :]) * 0.5
    # Normalize "height" so the slope vector has reasonable magnitude.
    scale = 8.0
    dzdx *= scale
    dzdy *= scale
    # Light direction (upper-left, slightly down): azimuth=315°, altitude=45°.
    az = math.radians(315.0)
    alt = math.radians(45.0)
    lx = math.cos(alt) * math.cos(az - math.pi / 2)
    ly = math.cos(alt) * math.sin(az - math.pi / 2)
    lz = math.sin(alt)
    # Normal = (-dzdx, -dzdy, 1) normalized.
    nx = -dzdx
    ny = -dzdy
    nz = np.ones_like(height)
    nlen = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx /= nlen; ny /= nlen; nz /= nlen
    shade = np.clip(nx * lx + ny * ly + nz * lz, 0, 1)
    # Boost contrast slightly.
    shade = shade ** 1.2
    # Tint by depth (combine with hypsometric for richness).
    depth_norm = np.clip((height - height.min()) / (np.percentile(height, 99) - height.min() + 1e-6), 0, 1)
    ti = 1 - depth_norm   # high = far
    r_arr = (30  + ti * (140 - 30))  * shade + 8 * (1 - shade)
    g_arr = (40  + ti * (180 - 40))  * shade + 10 * (1 - shade)
    b_arr = (60  + ti * (230 - 60))  * shade + 18 * (1 - shade)
    rgb = np.stack([np.clip(r_arr, 0, 255),
                    np.clip(g_arr, 0, 255),
                    np.clip(b_arr, 0, 255)], axis=-1).astype(np.uint8)
    from PIL import Image as PImg
    relief_img = PImg.fromarray(rgb, mode='RGB').convert("RGBA")
    im.alpha_composite(relief_img)

    _equipotential_overlay(draw, bodies, eps2, color=(255, 255, 255, 110))
    _draw_bodies(draw, bodies)
    _header(
        im,
        "Hillshade (shaded relief) + equipotential rings",
        "potential surface lit from upper-left → 3D-lit topographic feel on a flat 2D plane",
    )
    path = os.path.join(OUT_DIR, "mockup_hillshade.png")
    im.save(path)
    return path


def mockup_rubber_sheet():
    """All-in-one mockup: oblique 3D rubber-sheet + hillshade + hypsometric
    tint + equipotential rings + particle trails. Trying to match the
    user's description: '弹性橡皮膜 + 质量下凹 + 粒子流动 + 等势线 +
    曲面上直线'."""
    try:
        import numpy as np
    except ImportError:
        print("[rubber] numpy required, skipping.")
        return None
    im = Image.new("RGBA", (W, H), (5, 6, 12, 255))
    draw = ImageDraw.Draw(im, "RGBA")
    bodies = _OPT_BODIES
    eps2 = _OPT_EPS2

    # ── Hypsometric + hillshade combined base ──
    xs = np.arange(W).reshape(1, W)
    ys = np.arange(H).reshape(H, 1)
    phi = np.zeros((H, W), dtype=np.float64)
    for b in bodies:
        dx = xs - b["x"]
        dy = ys - b["y"]
        r = np.sqrt(dx * dx + dy * dy + eps2)
        phi += -b["m"] / r
    depth = -phi
    # Oblique Y-displacement for "rubber sheet sag" visual effect.
    # We render into a flat 2D buffer but tint+shade in a way that
    # SUGGESTS 3D sag (lower = darker, with slope shading).
    dzdx = np.zeros_like(depth)
    dzdy = np.zeros_like(depth)
    dzdx[:, 1:-1] = (depth[:, 2:] - depth[:, :-2]) * 0.5
    dzdy[1:-1, :] = (depth[2:, :] - depth[:-2, :]) * 0.5
    scale = 14.0
    dzdx *= scale; dzdy *= scale
    # Light: upper-left at 35° altitude. Stronger than basic hillshade.
    az = math.radians(315.0); alt = math.radians(35.0)
    lx = math.cos(alt) * math.cos(az - math.pi / 2)
    ly = math.cos(alt) * math.sin(az - math.pi / 2)
    lz = math.sin(alt)
    nx = -dzdx; ny = -dzdy; nz = np.ones_like(depth)
    nlen = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx /= nlen; ny /= nlen; nz /= nlen
    shade = np.clip(nx * lx + ny * ly + nz * lz, 0, 1) ** 1.4
    # Hypsometric ramp: deep = dark indigo, shallow = warm cyan
    lo, hi = np.percentile(depth, [3, 99])
    t = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    ti = 1 - t  # high = far (flat)
    # Mix base color: deep indigo → warm cool cyan
    r0 = 12  + ti * (60  - 12)
    g0 = 18  + ti * (120 - 18)
    b0 = 40  + ti * (180 - 40)
    # Multiply by shade for depth.
    r1 = (r0 * shade + 6  * (1 - shade) * 0.5) * 1.0
    g1 = (g0 * shade + 8  * (1 - shade) * 0.5) * 1.0
    b1 = (b0 * shade + 18 * (1 - shade) * 0.5) * 1.0
    rgb = np.stack([np.clip(r1, 0, 255),
                    np.clip(g1, 0, 255),
                    np.clip(b1, 0, 255)], axis=-1).astype(np.uint8)
    from PIL import Image as PImg
    base = PImg.fromarray(rgb, mode='RGB').convert("RGBA")
    im.alpha_composite(base)

    # ── Equipotential rings (elevation contours on the sag surface) ──
    _equipotential_overlay(draw, bodies, eps2, color=(220, 240, 255, 130))

    # ── "Curvilinear grid" via warped vertical-and-horizontal lines ──
    # Project flat (x,y) lines onto the sag surface by displacing y
    # downward by the depth (small fraction so it looks like an
    # oblique view of the rubber sheet without folds).
    GS = 60
    proj_scale = 0.18    # how much depth pulls grid lines visually
    rng = np.random.default_rng(42)
    for gy in range(0, H + 1, GS):
        pts = []
        for x in range(0, W + 1, 6):
            d = depth[min(H-1, gy), min(W-1, x)] - depth[0, 0]
            y_proj = gy + d * proj_scale
            pts.append((x, y_proj))
        draw.line(pts, fill=(180, 210, 240, 75), width=1)
    for gx in range(0, W + 1, GS):
        pts = []
        for y in range(0, H + 1, 6):
            d = depth[min(H-1, y), min(W-1, gx)] - depth[0, 0]
            y_proj = y + d * proj_scale
            pts.append((gx, y_proj))
        draw.line(pts, fill=(180, 210, 240, 75), width=1)

    # ── Particle trails (faint, suggestive of flow) ──
    NUM = 400
    LEN = 16
    STEP = 2.5
    halo = PImg.new("RGBA", (W, H), (0, 0, 0, 0))
    hdraw = ImageDraw.Draw(halo, "RGBA")
    for _ in range(NUM):
        px = rng.uniform(20, W - 20)
        py = rng.uniform(20, H - 20)
        # Skip if inside body
        skip = False
        for b in bodies:
            if (px - b["x"]) ** 2 + (py - b["y"]) ** 2 < (b["r"] + 5) ** 2:
                skip = True; break
        if skip: continue
        for k in range(LEN):
            ax = 0.0; ay = 0.0
            for b in bodies:
                dx = b["x"] - px; dy = b["y"] - py
                r2 = dx * dx + dy * dy + eps2
                r = math.sqrt(r2)
                kk = b["m"] / (r2 * r)
                ax += kk * dx; ay += kk * dy
            amag = math.sqrt(ax * ax + ay * ay) + 1e-9
            nx = px + ax / amag * STEP
            ny = py + ay / amag * STEP
            tail = (LEN - k) / LEN
            a = int(180 * tail)
            hdraw.line([(px, py), (nx, ny)], fill=(220, 240, 255, a), width=1)
            px, py = nx, ny
            if px < 0 or px > W or py < 0 or py > H: break
            inb = False
            for b in bodies:
                if (px - b["x"]) ** 2 + (py - b["y"]) ** 2 < b["r"] * b["r"]:
                    inb = True; break
            if inb: break
    im.alpha_composite(halo)

    # ── Bodies ──
    _draw_bodies(draw, bodies)
    _header(
        im,
        "Rubber sheet — depth tint + hillshade + curvilinear grid + particle trails + rings",
        "all-in-one combo trying to match: 弹性橡皮膜 + 质量下凹 + 粒子流动 + 等势线 + 曲面直线",
    )
    path = os.path.join(OUT_DIR, "mockup_rubber_sheet.png")
    im.save(path)
    return path


def _draw_topdown_variant(im, draw, bodies, eps2, H_CAM, SAG_DEPTH, GS, max_foreshort, label):
    """Render one variant of the top-down perspective mesh with given params."""
    import numpy as np
    cx_screen = W / 2
    cy_screen = H / 2

    def project(x, y, z):
        depth_from_cam = H_CAM - z * SAG_DEPTH
        foreshort = H_CAM / max(depth_from_cam, 0.05)
        if foreshort > max_foreshort:
            foreshort = max_foreshort
        sx = cx_screen + (x - cx_screen) * foreshort
        sy = cy_screen + (y - cy_screen) * foreshort
        return sx, sy

    pad = int(W * 0.3)
    cols = (W + 2 * pad) // GS + 1
    rows = (H + 2 * pad) // GS + 1
    xs_world = np.arange(cols) * GS - pad
    ys_world = np.arange(rows) * GS - pad
    phi = np.zeros((rows, cols), dtype=np.float64)
    for b in bodies:
        dx = xs_world.reshape(1, cols) - b["x"]
        dy = ys_world.reshape(rows, 1) - b["y"]
        r = np.sqrt(dx * dx + dy * dy + eps2)
        phi += -b["m"] / r
    phi_abs = np.abs(phi)
    phi_max = float(phi_abs.max())
    if phi_max < 1e-6: return
    z = np.clip(1.0 - phi_abs / phi_max, 0.0, 1.0)
    sx = np.zeros_like(z); sy = np.zeros_like(z)
    for j in range(rows):
        for i in range(cols):
            sx[j, i], sy[j, i] = project(xs_world[i], ys_world[j], z[j, i])
    for j in range(rows):
        for i in range(cols - 1):
            x0, y0 = sx[j, i], sy[j, i]
            x1, y1 = sx[j, i + 1], sy[j, i + 1]
            zavg = (z[j, i] + z[j, i + 1]) * 0.5
            bright = 0.25 + 0.75 * zavg
            r_col = int(120 * bright); g_col = int(180 * bright); b_col = int(240 * bright)
            a = int(160 * (0.5 + 0.5 * bright))
            draw.line([(x0, y0), (x1, y1)], fill=(r_col, g_col, b_col, a), width=1)
    for i in range(cols):
        for j in range(rows - 1):
            x0, y0 = sx[j, i], sy[j, i]
            x1, y1 = sx[j + 1, i], sy[j + 1, i]
            zavg = (z[j, i] + z[j + 1, i]) * 0.5
            bright = 0.25 + 0.75 * zavg
            r_col = int(120 * bright); g_col = int(180 * bright); b_col = int(240 * bright)
            a = int(160 * (0.5 + 0.5 * bright))
            draw.line([(x0, y0), (x1, y1)], fill=(r_col, g_col, b_col, a), width=1)
    _equipotential_overlay(draw, bodies, eps2, color=(220, 240, 255, 130))
    _draw_bodies(draw, bodies)
    draw.text((20, 20), label, font=FONT_CAPTION, fill=(240, 240, 250, 240))


def mockup_topdown_variants():
    """Three parameter variants of the top-down perspective mesh."""
    try:
        import numpy as np
    except ImportError:
        return None
    bodies = _OPT_BODIES
    eps2 = _OPT_EPS2
    # 2x2 grid of variants
    paths = []
    variants = [
        ("subtle: SAG=0.5 H=1.0 GS=16 maxF=3",  1.0, 0.5,  16, 3.0,  "topdown_subtle"),
        ("medium: SAG=0.7 H=1.0 GS=18 maxF=4",  1.0, 0.7,  18, 4.0,  "topdown_medium"),
        ("strong: SAG=0.85 H=1.0 GS=20 maxF=5", 1.0, 0.85, 20, 5.0,  "topdown_strong"),
        ("dense: SAG=0.6 H=1.0 GS=12 maxF=3.5", 1.0, 0.6,  12, 3.5,  "topdown_dense"),
    ]
    for label, h_cam, sag, gs, max_f, name in variants:
        im = Image.new("RGBA", (W, H), (5, 6, 12, 255))
        draw = ImageDraw.Draw(im, "RGBA")
        _draw_topdown_variant(im, draw, bodies, eps2, h_cam, sag, gs, max_f, label)
        _header(im, "Top-down perspective mesh — variant",
                "(see label inside canvas for SAG/H_CAM/GS/maxForeshort)")
        path = os.path.join(OUT_DIR, f"mockup_{name}.png")
        im.save(path)
        paths.append(path)
    return paths


def mockup_topdown_perspective():
    """User spec (verbatim): 3D potential surface where z = 1 - |φ|/|φ_max|,
    body centers at z=0 (deepest valley) and viewport edge at z=1 (plateau).
    Mesh this surface, project from directly above using standard perspective.
    Grid cells near bodies get visually pulled inward toward the body (because
    those vertices are 'further from the camera'). Mesh extends to viewport
    edge (borderless)."""
    try:
        import numpy as np
    except ImportError:
        print("[topdown] numpy required, skipping.")
        return None
    im = Image.new("RGBA", (W, H), (5, 6, 12, 255))
    draw = ImageDraw.Draw(im, "RGBA")
    bodies = _OPT_BODIES
    eps2 = _OPT_EPS2

    cx_screen = W / 2
    cy_screen = H / 2

    # Perspective parameters: camera at height H_cam above ground.
    # foreshortening factor for vertex at elevation z:
    #   factor = H_cam / (H_cam - z * SAG_DEPTH)
    # z=1 (plateau, near camera) → factor close to H_cam/(H_cam - SAG_DEPTH)
    # z=0 (valley, far from camera) → factor = 1 (no foreshortening change)
    # Larger SAG_DEPTH → more dramatic perspective.
    H_CAM = 1.0
    SAG_DEPTH = 0.85   # how deep the valleys go (0..1)

    def project(x, y, z):
        # Higher z = closer to camera = MORE foreshortening (spread out).
        # Lower z = farther from camera = LESS foreshortening (compressed in).
        depth_from_cam = H_CAM - z * SAG_DEPTH
        foreshort = H_CAM / max(depth_from_cam, 0.05)
        sx = cx_screen + (x - cx_screen) * foreshort
        sy = cy_screen + (y - cy_screen) * foreshort
        return sx, sy

    # Compute per-vertex z on an expanded grid (overshoot viewport so the
    # projected mesh reliably covers the visible canvas).
    GS = 18
    pad = int(W * 0.4)   # extra cells outside the visible viewport
    cols = (W + 2 * pad) // GS + 1
    rows = (H + 2 * pad) // GS + 1
    xs_world = np.arange(cols) * GS - pad
    ys_world = np.arange(rows) * GS - pad

    # phi at each vertex.
    phi = np.zeros((rows, cols), dtype=np.float64)
    for b in bodies:
        dx = xs_world.reshape(1, cols) - b["x"]
        dy = ys_world.reshape(rows, 1) - b["y"]
        r = np.sqrt(dx * dx + dy * dy + eps2)
        phi += -b["m"] / r
    # Normalize: z = 1 - |phi|/|phi_max|
    phi_abs = np.abs(phi)
    phi_max = float(phi_abs.max())
    if phi_max < 1e-6:
        return None
    z = np.clip(1.0 - phi_abs / phi_max, 0.0, 1.0)

    # Project each vertex.
    sx = np.zeros_like(z)
    sy = np.zeros_like(z)
    for j in range(rows):
        for i in range(cols):
            sx[j, i], sy[j, i] = project(xs_world[i], ys_world[j], z[j, i])

    # Render mesh — horizontal then vertical lines.
    # Color modulated by z (deeper = darker; shallower = bright cyan).
    for j in range(rows):
        for i in range(cols - 1):
            x0, y0 = sx[j, i],     sy[j, i]
            x1, y1 = sx[j, i + 1], sy[j, i + 1]
            zavg = (z[j, i] + z[j, i + 1]) * 0.5
            # Brightness: deep valleys = dark; plateau = bright cyan
            bright = 0.25 + 0.75 * zavg
            r_col = int(120 * bright)
            g_col = int(180 * bright)
            b_col = int(240 * bright)
            a = int(150 * (0.4 + 0.6 * bright))
            draw.line([(x0, y0), (x1, y1)], fill=(r_col, g_col, b_col, a), width=1)
    for i in range(cols):
        for j in range(rows - 1):
            x0, y0 = sx[j, i],     sy[j, i]
            x1, y1 = sx[j + 1, i], sy[j + 1, i]
            zavg = (z[j, i] + z[j + 1, i]) * 0.5
            bright = 0.25 + 0.75 * zavg
            r_col = int(120 * bright)
            g_col = int(180 * bright)
            b_col = int(240 * bright)
            a = int(150 * (0.4 + 0.6 * bright))
            draw.line([(x0, y0), (x1, y1)], fill=(r_col, g_col, b_col, a), width=1)

    # Equipotential rings on top (in world space, NOT projected through
    # perspective — they stay as clean topographic rings around bodies).
    _equipotential_overlay(draw, bodies, eps2, color=(220, 240, 255, 130))
    _draw_bodies(draw, bodies)
    _header(
        im,
        "Top-down perspective mesh — z=1-|φ|/|φ_max|, project from above",
        "valleys compressed inward, plateaus spread out; mesh extends past viewport edges",
    )
    path = os.path.join(OUT_DIR, "mockup_topdown_perspective.png")
    im.save(path)
    return path


def mockup_ortho_topdown():
    """Orthographic top-down view (user spec: 从 z 轴上方正交投影).
    Mesh has no projection deformation — (x,y) stays put — so the
    'curvature' must be conveyed via per-pixel shading + color + contour
    rings + particles. This is the classic topographic-map approach.
    No grid wireframe (it'd project to a flat Cartesian grid, useless)."""
    try:
        import numpy as np
    except ImportError:
        return None
    im = Image.new("RGBA", (W, H), (5, 6, 12, 255))
    bodies = _OPT_BODIES
    eps2 = _OPT_EPS2

    # Pixel-level φ and depth = -φ.
    xs = np.arange(W).reshape(1, W)
    ys = np.arange(H).reshape(H, 1)
    phi = np.zeros((H, W), dtype=np.float64)
    for b in bodies:
        dx = xs - b["x"]; dy = ys - b["y"]
        r = np.sqrt(dx * dx + dy * dy + eps2)
        phi += -b["m"] / r
    depth = -phi
    # Normalize: z = 1 - |φ|/|φ_max| (per user spec).
    phi_max = float(np.abs(phi).max())
    z = np.clip(1.0 - np.abs(phi) / phi_max, 0.0, 1.0)

    # Surface gradient (treat z as elevation: high = plateau, low = valley).
    # But wait — for "rubber sheet" we want the SURFACE = -φ (deep at bodies).
    # So use surface_z = depth normalized so valleys ARE valleys.
    surf = depth - depth.min()
    surf = surf / max(surf.max(), 1e-6) * 80   # in px-equivalent units
    dzdx = np.zeros_like(surf)
    dzdy = np.zeros_like(surf)
    dzdx[:, 1:-1] = (surf[:, 2:] - surf[:, :-2]) * 0.5
    dzdy[1:-1, :] = (surf[2:, :] - surf[:-2, :]) * 0.5

    # Strong light direction: low altitude angle for dramatic shadows.
    az = math.radians(315.0); alt = math.radians(28.0)
    lx = math.cos(alt) * math.cos(az - math.pi / 2)
    ly = math.cos(alt) * math.sin(az - math.pi / 2)
    lz = math.sin(alt)
    nx = -dzdx; ny = -dzdy; nz = np.ones_like(surf) * 1.0
    nlen = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx /= nlen; ny /= nlen; nz /= nlen
    # Surface inverted (rubber sheet sags DOWN at bodies, so normal flips).
    shade = np.clip(-(nx * lx + ny * ly) + nz * lz, 0, 1) ** 1.3
    # Bias upward so darkest shadow isn't pitch-black.
    shade = 0.15 + 0.85 * shade

    # Hypsometric: deep navy in valleys → bright cool cyan on plateau.
    ti = z  # high z (1) = plateau bright, low z (0) = valley dark
    r_arr = 8   + ti * (130 - 8)
    g_arr = 14  + ti * (180 - 14)
    b_arr = 35  + ti * (240 - 35)
    # Modulate by hillshade.
    r1 = r_arr * shade
    g1 = g_arr * shade
    b1 = b_arr * shade
    rgb = np.stack([np.clip(r1, 0, 255),
                    np.clip(g1, 0, 255),
                    np.clip(b1, 0, 255)], axis=-1).astype(np.uint8)
    from PIL import Image as PImg
    base = PImg.fromarray(rgb, mode='RGB').convert("RGBA")
    im.alpha_composite(base)

    # Particle trails (motion overlay).
    draw = ImageDraw.Draw(im, "RGBA")
    rng = np.random.default_rng(7)
    NUM = 350; LEN = 18; STEP = 2.5
    halo = PImg.new("RGBA", (W, H), (0, 0, 0, 0))
    hdraw = ImageDraw.Draw(halo, "RGBA")
    for _ in range(NUM):
        px = rng.uniform(20, W - 20); py = rng.uniform(20, H - 20)
        skip = False
        for b in bodies:
            if (px - b["x"]) ** 2 + (py - b["y"]) ** 2 < (b["r"] + 5) ** 2:
                skip = True; break
        if skip: continue
        for k in range(LEN):
            ax = 0.0; ay = 0.0
            for b in bodies:
                dx = b["x"] - px; dy = b["y"] - py
                r2 = dx * dx + dy * dy + eps2
                r = math.sqrt(r2)
                kk = b["m"] / (r2 * r)
                ax += kk * dx; ay += kk * dy
            amag = math.sqrt(ax * ax + ay * ay) + 1e-9
            nx_p = px + ax / amag * STEP
            ny_p = py + ay / amag * STEP
            tail = (LEN - k) / LEN
            a = int(200 * tail)
            hdraw.line([(px, py), (nx_p, ny_p)], fill=(230, 245, 255, a), width=1)
            px, py = nx_p, ny_p
            if px < 0 or px > W or py < 0 or py > H: break
            inb = False
            for b in bodies:
                if (px - b["x"]) ** 2 + (py - b["y"]) ** 2 < b["r"] * b["r"]:
                    inb = True; break
            if inb: break
    im.alpha_composite(halo)

    # Equipotential rings on top.
    draw = ImageDraw.Draw(im, "RGBA")
    _equipotential_overlay(draw, bodies, eps2, color=(255, 255, 255, 150))
    _draw_bodies(draw, bodies)
    _header(
        im,
        "Orthographic top-down — hypsometric tint + hillshade + rings + particles",
        "正交投影 from z-axis. No mesh wireframe (would project flat). Depth shown via shading+color+rings.",
    )
    path = os.path.join(OUT_DIR, "mockup_ortho_topdown.png")
    im.save(path)
    return path


def _draw_oblique_variant(im, draw, bodies, eps2, tilt_deg, max_sag, GS, light_alt_deg, label):
    """One oblique-3D variant with given params. Mesh auto-fills the canvas."""
    import numpy as np
    cy_factor = math.cos(math.radians(tilt_deg))
    z_factor  = math.sin(math.radians(tilt_deg))
    pad_x = int(W * 0.05)
    # Y padding needs to cover both the world extent compressed by cy_factor
    # AND the sag shift. Compute generously so mesh always covers canvas.
    needed_world_y = H / cy_factor + max_sag / cy_factor
    pad_y_total = max(0, int(needed_world_y - H))
    pad_y_top = pad_y_total // 2
    pad_y_bot = pad_y_total - pad_y_top
    cols = (W + 2 * pad_x) // GS + 1
    rows = (H + pad_y_top + pad_y_bot) // GS + 1
    xs_world = np.arange(cols) * GS - pad_x
    ys_world = np.arange(rows) * GS - pad_y_top

    phi = np.zeros((rows, cols), dtype=np.float64)
    for b in bodies:
        dx = xs_world.reshape(1, cols) - b["x"]
        dy = ys_world.reshape(rows, 1) - b["y"]
        r = np.sqrt(dx * dx + dy * dy + eps2)
        phi += -b["m"] / r
    depth = -phi
    sag_norm = (depth - depth.min()) / max(depth.max() - depth.min(), 1e-6)
    sag = sag_norm * max_sag

    sx = xs_world.reshape(1, cols) + np.zeros((rows, cols))
    sy = ys_world.reshape(rows, 1) * cy_factor + sag * z_factor
    # Shift so mesh fills canvas vertically.
    sy_min = sy.min()
    sy_max = sy.max()
    target_min = 0
    target_max = H
    if sy_max - sy_min > 1e-6:
        scale = (target_max - target_min) / (sy_max - sy_min)
        sy = (sy - sy_min) * scale + target_min
        # NOTE: rescaling sy here also rescales the apparent sag, but we
        # accept that — the param max_sag already controls the relative
        # depth visually.

    dzdx = np.zeros_like(sag); dzdy = np.zeros_like(sag)
    dzdx[:, 1:-1] = (sag[:, 2:] - sag[:, :-2]) * 0.5
    dzdy[1:-1, :] = (sag[2:, :] - sag[:-2, :]) * 0.5
    az = math.radians(315.0); alt = math.radians(light_alt_deg)
    lx = math.cos(alt) * math.cos(az - math.pi / 2)
    ly = math.cos(alt) * math.sin(az - math.pi / 2)
    lz = math.sin(alt)
    nx = dzdx; ny = dzdy; nz = np.ones_like(sag)
    nlen = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx /= nlen; ny /= nlen; nz /= nlen
    shade = np.clip(nx * lx + ny * ly + nz * lz, 0, 1) ** 1.2
    shade = 0.18 + 0.82 * shade

    valley = sag_norm
    plateau = 1.0 - valley
    r_arr = 15 + plateau * (110 - 15)
    g_arr = 22 + plateau * (165 - 22)
    b_arr = 50 + plateau * (220 - 50)
    cell_r = (r_arr[:-1,:-1] + r_arr[:-1,1:] + r_arr[1:,:-1] + r_arr[1:,1:]) * 0.25
    cell_g = (g_arr[:-1,:-1] + g_arr[:-1,1:] + g_arr[1:,:-1] + g_arr[1:,1:]) * 0.25
    cell_b = (b_arr[:-1,:-1] + b_arr[:-1,1:] + b_arr[1:,:-1] + b_arr[1:,1:]) * 0.25
    cell_s = (shade[:-1,:-1] + shade[:-1,1:] + shade[1:,:-1] + shade[1:,1:]) * 0.25
    cell_r *= cell_s; cell_g *= cell_s; cell_b *= cell_s

    cell_indices = []
    for j in range(rows - 1):
        for i in range(cols - 1):
            cell_indices.append((ys_world[j] + ys_world[j+1], j, i))
    cell_indices.sort(key=lambda t: t[0])

    for (_, j, i) in cell_indices:
        x0, y0 = sx[j, i],     sy[j, i]
        x1, y1 = sx[j, i+1],   sy[j, i+1]
        x2, y2 = sx[j+1, i+1], sy[j+1, i+1]
        x3, y3 = sx[j+1, i],   sy[j+1, i]
        rcol = int(np.clip(cell_r[j, i], 0, 255))
        gcol = int(np.clip(cell_g[j, i], 0, 255))
        bcol = int(np.clip(cell_b[j, i], 0, 255))
        draw.polygon([(x0,y0),(x1,y1),(x2,y2),(x3,y3)],
                     fill=(rcol, gcol, bcol, 255),
                     outline=(min(255, rcol+30), min(255, gcol+30), min(255, bcol+30), 255))

    # Bodies in world space → project to screen.
    for b in bodies:
        bj = int(np.clip((b["y"] + pad_y_top) / GS, 0, rows - 1))
        bi = int(np.clip((b["x"] + pad_x) / GS, 0, cols - 1))
        bsx = sx[bj, bi]
        bsy = sy[bj, bi]
        for k in range(6, 0, -1):
            rr = b["r"] + k * 4
            draw.ellipse([bsx - rr, bsy - rr, bsx + rr, bsy + rr],
                         fill=(*b["color"], max(0, 30 - k * 4)))
        draw.ellipse([bsx - b["r"], bsy - b["r"], bsx + b["r"], bsy + b["r"]],
                     fill=(*b["color"], 255),
                     outline=(255, 255, 255, 160), width=2)

    draw.text((20, 20), label, font=FONT_CAPTION, fill=(245, 245, 250, 240))


def mockup_oblique_variants():
    """Four tilt-angle variants of dense oblique-3D mesh — user wants to
    see which tilt makes 2D bodies best aligned with their sag wells."""
    try:
        import numpy as np
    except ImportError:
        return None
    paths = []
    bodies = _OPT_BODIES
    eps2 = _OPT_EPS2
    variants = [
        ("tilt=45° dense (SAG=140 GS=14)",  45, 140, 14, 40, "oblique_tilt45"),
        ("tilt=55° dense (SAG=140 GS=14)",  55, 140, 14, 38, "oblique_tilt55"),
        ("tilt=60° dense (SAG=140 GS=14)",  60, 140, 14, 35, "oblique_tilt60"),
        ("tilt=70° dense (SAG=140 GS=14)",  70, 140, 14, 32, "oblique_tilt70"),
    ]
    for label, tilt, sag, gs, lalt, name in variants:
        im = Image.new("RGBA", (W, H), (5, 6, 12, 255))
        draw = ImageDraw.Draw(im, "RGBA")
        _draw_oblique_variant(im, draw, bodies, eps2, tilt, sag, gs, lalt, label)
        _header(im, "Oblique 3D rubber sheet — variant", "(label inside canvas)")
        path = os.path.join(OUT_DIR, f"mockup_{name}.png")
        im.save(path)
        paths.append(path)
    return paths


def mockup_oblique_3d():
    """Oblique 30° view of the sagging 3D potential surface, with
    painter's-algorithm occlusion so back grid lines DON'T show
    through front ones. User spec: 'from z-axis 30° angle, see
    "往下凹的感觉" at masses, no 网格透过网格'."""
    try:
        import numpy as np
    except ImportError:
        return None
    im = Image.new("RGBA", (W, H), (5, 6, 12, 255))
    draw = ImageDraw.Draw(im, "RGBA")
    bodies = _OPT_BODIES
    eps2 = _OPT_EPS2

    # Camera tilt: 30° from z-axis = 60° elevation above ground plane.
    # Equivalent projection:
    #   screen_x = world_x
    #   screen_y = world_y * cos(60°) + sag * sin(60°)
    # where sag = how deep the surface drops at this (x, y).
    # Positive sag → screen_y shifts DOWN (visually deeper).
    TILT_DEG = 60.0
    cy_factor = math.cos(math.radians(TILT_DEG))      # y compression
    z_factor  = math.sin(math.radians(TILT_DEG))      # z → screen y shift

    # Build grid. Extends past viewport in y so back rim stays visible
    # when surface tilts; x not extended much because tilt is only in y.
    GS = 22
    pad_x = int(W * 0.05)
    pad_y_top = int(H * 0.40)   # extra room above for warped far edge
    pad_y_bot = int(H * 0.20)
    cols = (W + 2 * pad_x) // GS + 1
    rows = (H + pad_y_top + pad_y_bot) // GS + 1
    xs_world = np.arange(cols) * GS - pad_x
    ys_world = np.arange(rows) * GS - pad_y_top

    # Compute φ → sag depth (deeper at bodies).
    phi = np.zeros((rows, cols), dtype=np.float64)
    for b in bodies:
        dx = xs_world.reshape(1, cols) - b["x"]
        dy = ys_world.reshape(rows, 1) - b["y"]
        r = np.sqrt(dx * dx + dy * dy + eps2)
        phi += -b["m"] / r
    depth = -phi
    # Normalize sag to [0, MAX_SAG] in screen-y units.
    MAX_SAG = 120.0
    sag_norm = (depth - depth.min()) / max(depth.max() - depth.min(), 1e-6)
    sag = sag_norm * MAX_SAG    # px-equivalent

    # Project vertices.
    sx = xs_world.reshape(1, cols) + np.zeros((rows, cols))
    sy = ys_world.reshape(rows, 1) * cy_factor + sag * z_factor

    # For shading: compute surface gradient on the world surface.
    dzdx = np.zeros_like(sag); dzdy = np.zeros_like(sag)
    dzdx[:, 1:-1] = (sag[:, 2:] - sag[:, :-2]) * 0.5
    dzdy[1:-1, :] = (sag[2:, :] - sag[:-2, :]) * 0.5
    # Light from upper-left, somewhat steep.
    az = math.radians(315.0); alt = math.radians(40.0)
    lx = math.cos(alt) * math.cos(az - math.pi / 2)
    ly = math.cos(alt) * math.sin(az - math.pi / 2)
    lz = math.sin(alt)
    nx = dzdx; ny = dzdy; nz = np.ones_like(sag)
    nlen = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx /= nlen; ny /= nlen; nz /= nlen
    shade = np.clip(nx * lx + ny * ly + nz * lz, 0, 1) ** 1.2
    shade = 0.18 + 0.82 * shade

    # Hypsometric tint: deep valley = dark navy, plateau = warm cyan.
    valley = sag_norm   # 0 = flat, 1 = deepest
    plateau = 1.0 - valley
    r_arr = 15 + plateau * (110 - 15)
    g_arr = 22 + plateau * (165 - 22)
    b_arr = 50 + plateau * (220 - 50)

    # Per-cell color (avg of corners).
    cell_r = (r_arr[:-1, :-1] + r_arr[:-1, 1:] + r_arr[1:, :-1] + r_arr[1:, 1:]) * 0.25
    cell_g = (g_arr[:-1, :-1] + g_arr[:-1, 1:] + g_arr[1:, :-1] + g_arr[1:, 1:]) * 0.25
    cell_b = (b_arr[:-1, :-1] + b_arr[:-1, 1:] + b_arr[1:, :-1] + b_arr[1:, 1:]) * 0.25
    cell_shade = (shade[:-1, :-1] + shade[:-1, 1:] + shade[1:, :-1] + shade[1:, 1:]) * 0.25
    cell_r *= cell_shade
    cell_g *= cell_shade
    cell_b *= cell_shade

    # Sort cells by depth (world_y) — back to front for painter's algorithm.
    # Use the cell's NORMALIZED-Y so far-cells (small y) get drawn first.
    cell_indices = []
    for j in range(rows - 1):
        for i in range(cols - 1):
            # Cell back-most edge (smallest world_y = top of canvas).
            avg_world_y = (ys_world[j] + ys_world[j + 1]) * 0.5
            # Sort key: smaller world_y = farther = drawn first.
            cell_indices.append((avg_world_y, j, i))
    cell_indices.sort(key=lambda t: t[0])

    # Render each cell as a filled quad with thin wireframe edges.
    for (_, j, i) in cell_indices:
        x0, y0 = sx[j,   i],     sy[j,   i]
        x1, y1 = sx[j,   i + 1], sy[j,   i + 1]
        x2, y2 = sx[j+1, i + 1], sy[j+1, i + 1]
        x3, y3 = sx[j+1, i],     sy[j+1, i]
        rcol = int(np.clip(cell_r[j, i], 0, 255))
        gcol = int(np.clip(cell_g[j, i], 0, 255))
        bcol = int(np.clip(cell_b[j, i], 0, 255))
        # Filled quad (alpha 255 so back cells are FULLY hidden by front).
        draw.polygon([(x0, y0), (x1, y1), (x2, y2), (x3, y3)],
                     fill=(rcol, gcol, bcol, 255),
                     outline=(rcol + 35, gcol + 35, bcol + 35, 255))

    # Bodies — project to screen, then draw on top.
    for b in bodies:
        # Find the sag at body center.
        bj = int(np.clip((b["y"] + pad_y_top) / GS, 0, rows - 1))
        bi = int(np.clip((b["x"] + pad_x) / GS, 0, cols - 1))
        body_sag = sag[bj, bi]
        bsx = b["x"]
        bsy = b["y"] * cy_factor + body_sag * z_factor
        for k in range(6, 0, -1):
            rr = b["r"] + k * 4
            draw.ellipse([bsx - rr, bsy - rr, bsx + rr, bsy + rr],
                         fill=(*b["color"], max(0, 30 - k * 4)))
        draw.ellipse([bsx - b["r"], bsy - b["r"], bsx + b["r"], bsy + b["r"]],
                     fill=(*b["color"], 255),
                     outline=(255, 255, 255, 160), width=2)

    _header(
        im,
        "Oblique 30° — 3D rubber sheet view with painter's-algorithm occlusion",
        "valleys visibly sag downward in screen-y; back cells hidden by front cells (no 网格透过网格)",
    )
    path = os.path.join(OUT_DIR, "mockup_oblique_3d.png")
    im.save(path)
    return path


if __name__ == "__main__":
    p1 = mockup_3d()
    p2 = mockup_2d()
    p3 = mockup_flow()
    p4 = mockup_mix()
    p5 = mockup_wrap()
    p6 = mockup_hard_vs_smooth()
    p7 = mockup_cluster_depth()
    p8 = mockup_option3()
    p9 = mockup_option4()
    p10 = mockup_lic()
    p11 = mockup_hypsometric()
    p12 = mockup_hillshade()
    p13 = mockup_rubber_sheet()
    p14 = mockup_topdown_perspective()
    p15_list = mockup_topdown_variants()
    p16 = mockup_ortho_topdown()
    p17 = mockup_oblique_3d()
    p18_list = mockup_oblique_variants()
    print("Wrote:")
    for p in (p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14):
        if p:
            print(" ", p)
    if p15_list:
        for p in p15_list:
            print(" ", p)
    if p16: print(" ", p16)
    if p17: print(" ", p17)
    if p18_list:
        for p in p18_list:
            print(" ", p)
