"""
world_viewer.py — World Engine layer console (M0–M2 prototype)

Interactive interface on top of worldgen.py. The world stays a PURE FUNCTION:
    world(seed, x, y, t) -> layers
The static layers (elevation, moisture, D8 flow) come straight from worldgen.py
and are rebuilt only when seed/resolution changes. Everything time-dependent
(M2) is recomputed every frame, cheap and vectorized:

    day/night   — the sun sweeps one full longitude per sim day
    seasons     — sinusoidal temperature offset, opposite in each hemisphere
    tides       — sea level breathes with sin(t)

Interface:
    - big map + 7 layer thumbnails, all animating; CLICK a thumbnail to
      promote that layer to the main view
    - sliders: sea level, river threshold, season amp, tide amp,
      day/night depth, sim speed
    - seed textbox + random button; play/pause; reset to day 0
    - EXPORT button: writes a PNG sequence of the current view layer at the
      current seed/settings to exports/, plus a meta.json of every parameter

Run:  python3 world_viewer.py [--seed 42] [--size 256]
      (512 looks better, 256 animates faster; world rebuild takes a few
       seconds at 512 because flow accumulation is a Python loop)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox

from worldgen import elevation_field, moisture_field, compute_rivers, BIOME_COLORS

# ---------------------------------------------------------------------------
# Time model (M2). t is measured in sim DAYS.
# ---------------------------------------------------------------------------
YEAR_DAYS = 96.0          # one year = 96 days
TIDE_PERIOD = 0.52        # ~semi-diurnal tide

# ---------------------------------------------------------------------------
# Vectorized biome classification -> int ids (fast enough to run every frame;
# same thresholds as worldgen.classify_biomes, which uses object arrays).
# ---------------------------------------------------------------------------
BIOME_NAMES = list(BIOME_COLORS.keys())
BIOME_LUT = np.array([BIOME_COLORS[n] for n in BIOME_NAMES], dtype=np.float32)
BID = {n: i for i, n in enumerate(BIOME_NAMES)}


def biome_ids(e, t, m, sea):
    water = e < sea
    conds = [
        water & (e < sea - 0.12),
        water & (e >= sea - 0.05),
        water,
        e > 0.92,
        e > 0.82,
        e < sea + 0.015,
        t < 0.2,
        t < 0.35,
        t < 0.5,
        m < 0.25,
        m < 0.45,
        m < 0.65,
        m < 0.82,
    ]
    choices = [BID[n] for n in (
        "deep_ocean", "shallow", "ocean", "high_peak", "mountain", "beach",
        "snow", "tundra", "taiga", "desert", "savanna", "grassland", "forest")]
    return np.select(conds, choices, default=BID["jungle"]).astype(np.int16)


# ---------------------------------------------------------------------------
# Time-dependent fields
# ---------------------------------------------------------------------------
def temperature_t(elev, lat, lat_signed, sea_eff, season_off):
    t = lat + season_off * lat_signed - np.clip(elev - sea_eff, 0, None) * 0.9
    return np.clip(t, 0, 1)


def daylight_row(xn, sun_x, depth):
    """Per-longitude light factor: 1 at noon, floor at midnight."""
    c = np.cos(2 * np.pi * (xn - sun_x))
    s = np.clip((c + 0.15) / 0.45, 0, 1)
    s = s * s * (3 - 2 * s)
    floor = 1 - 0.72 * depth
    return floor + (1 - floor) * s


def color_ramp(v, stops, colors):
    """v in [0,1] -> RGB via piecewise-linear ramp. colors: (k,3)."""
    colors = np.asarray(colors, dtype=np.float32)
    return np.stack([np.interp(v, stops, colors[:, c]) for c in range(3)], axis=-1)


TMP_RAMP = ([0.0, 0.5, 1.0], [(58, 103, 196), (232, 228, 216), (200, 80, 46)])
MOI_RAMP = ([0.0, 0.5, 1.0], [(201, 163, 90), (214, 214, 196), (62, 143, 122)])
FLOW_RAMP = ([0.0, 0.5, 1.0], [(12, 18, 30), (32, 110, 150), (180, 235, 250)])

LAYERS = [
    ("composite", "World"),
    ("elevation", "Elevation"),
    ("temperature", "Temperature"),
    ("moisture", "Moisture"),
    ("flow", "Flow"),
    ("biome", "Biomes"),
    ("light", "Daylight"),
]

# console theme
C_BG, C_PANEL, C_INK, C_MUTED, C_ACCENT = "#0f1317", "#151b21", "#e8edf2", "#93a1af", "#e2a54f"


class WorldSlice:
    """Static per-resolution data + grids (full res, and strided for thumbs)."""

    def __init__(self, elev, moist, accum):
        self.elev, self.moist, self.accum = elev, moist, accum
        size = elev.shape[0]
        yn = (np.arange(size, dtype=np.float32) / size)[:, None]
        self.lat = 1 - np.abs(yn - 0.5) * 2
        self.lat_signed = (0.5 - yn) * 2
        self.xn = np.arange(size, dtype=np.float32) / size
        gy, gx = np.gradient(elev)
        self.shade = np.clip(1 - (gx + gy) * 2.2, 0.75, 1.25).astype(np.float32)
        self.log_accum = np.log1p(accum) / np.log1p(accum.max())

    def strided(self, st):
        s = WorldSlice.__new__(WorldSlice)
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                setattr(s, k, v[::st, ::st] if v.ndim == 2 else v[::st])
        return s


class Console:
    def __init__(self, seed=42, size=256):
        self.seed, self.size = seed, size
        self.t = 0.0
        self.playing = True
        self.layer = "composite"
        # river threshold tuned for 512²; scale down for smaller worlds
        self.sea_level = 0.42
        self.river_thr = round(350 * (size / 512) ** 1.5)
        self.season_amp, self.tide_amp = 0.18, 0.012
        self.day_night, self.speed = 0.65, 0.35
        self.exporting = False

        self.build_world()
        self.build_ui()

    # ---------------- world (heavy, on seed/size change only) ----------------
    def build_world(self):
        t0 = time.time()
        print(f"generating world seed={self.seed} size={self.size} ...", flush=True)
        elev = elevation_field(self.size, self.seed).astype(np.float32)
        moist = moisture_field(self.size, self.seed).astype(np.float32)
        _, accum = compute_rivers(elev, self.sea_level)
        self.full = WorldSlice(elev, moist, accum.astype(np.float32))
        st = max(1, self.size // 96)
        self.thumb = self.full.strided(st)
        print(f"  done in {time.time() - t0:.1f}s")

    # ---------------- pure render: (slice, layer, t) -> uint8 RGB ------------
    def frame_params(self, t):
        sea_eff = self.sea_level + self.tide_amp * np.sin(2 * np.pi * t / TIDE_PERIOD)
        season_off = self.season_amp * np.sin(2 * np.pi * t / YEAR_DAYS)
        sun_x = t % 1.0
        return sea_eff, season_off, sun_x

    def render(self, ws, layer, t):
        sea_eff, season_off, sun_x = self.frame_params(t)
        e = ws.elev

        if layer == "elevation":
            img = np.empty(e.shape + (3,), np.float32)
            sea_m = e < sea_eff
            f = np.clip(e / max(sea_eff, 1e-6), 0, 1)
            img[..., 0] = 16 + 70 * f
            img[..., 1] = 34 + 106 * f
            img[..., 2] = 78 + 108 * f
            g = np.clip((e - sea_eff) / max(1 - sea_eff, 1e-6), 0, 1)
            land_rgb = color_ramp(g, [0.0, 0.55, 1.0],
                                  [(88, 140, 80), (168, 150, 96), (245, 245, 248)])
            img[~sea_m] = land_rgb[~sea_m]
        elif layer == "temperature":
            tf = temperature_t(e, ws.lat, ws.lat_signed, sea_eff, season_off)
            img = color_ramp(tf, *TMP_RAMP)
        elif layer == "moisture":
            img = color_ramp(ws.moist, *MOI_RAMP)
        elif layer == "flow":
            img = color_ramp(ws.log_accum, *FLOW_RAMP)
        elif layer == "light":
            l = daylight_row(ws.xn, sun_x, self.day_night)[None, :]
            img = np.empty(e.shape + (3,), np.float32)
            img[..., 0] = 255 * l
            img[..., 1] = 248 * l
            img[..., 2] = 80 + 145 * l
        else:  # biome / composite
            tf = temperature_t(e, ws.lat, ws.lat_signed, sea_eff, season_off)
            ids = biome_ids(e, tf, ws.moist, sea_eff)
            img = BIOME_LUT[ids]
            if layer == "composite":
                land = e >= sea_eff
                img = img.copy()
                img[land] *= ws.shade[land, None]
                rivers = (ws.accum > self.river_thr) & land
                img[rivers] = (70, 130, 200)
                l = daylight_row(ws.xn, sun_x, self.day_night)[None, :]
                img[..., 0] *= l
                img[..., 1] *= l * 0.96 + 0.04
                img[..., 2] *= l * 0.82 + 0.18
        return np.clip(img, 0, 255).astype(np.uint8)

    # ---------------- UI ----------------
    def build_ui(self):
        plt.rcParams.update({
            "font.family": "monospace", "text.color": C_INK,
            "axes.edgecolor": C_MUTED, "xtick.color": C_MUTED, "ytick.color": C_MUTED,
        })
        self.fig = plt.figure(figsize=(14, 8.2), facecolor=C_BG)
        self.fig.canvas.manager.set_window_title("World Engine — layer console")

        # main map
        self.ax_main = self.fig.add_axes([0.03, 0.30, 0.46, 0.64])
        self.ax_main.set_facecolor(C_BG)
        self.ax_main.set_xticks([]), self.ax_main.set_yticks([])
        self.im_main = self.ax_main.imshow(self.render(self.full, self.layer, self.t))
        self.title = self.ax_main.set_title("", color=C_MUTED, fontsize=9, loc="left", pad=8)

        # thumbnails row
        self.thumb_axes, self.thumb_ims, self.thumb_titles = {}, {}, {}
        tw = 0.062
        for i, (key, name) in enumerate(LAYERS):
            ax = self.fig.add_axes([0.03 + i * (tw + 0.008), 0.155, tw, 0.11])
            ax.set_xticks([]), ax.set_yticks([])
            self.thumb_ims[key] = ax.imshow(self.render(self.thumb, key, self.t))
            self.thumb_titles[key] = ax.set_title(name.upper(), fontsize=6.5, color=C_MUTED, pad=3)
            self.thumb_axes[key] = ax
        self._style_thumb_sel()
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        # ---- right panel: sliders ----
        def slider(y, label, vmin, vmax, vinit, step=None, fmt="%.2f"):
            ax = self.fig.add_axes([0.60, y, 0.30, 0.022], facecolor=C_PANEL)
            s = Slider(ax, label + "  ", vmin, vmax, valinit=vinit,
                       valstep=step, color=C_ACCENT, valfmt=fmt)
            s.label.set_color(C_INK), s.label.set_fontsize(8.5)
            s.valtext.set_color(C_MUTED), s.valtext.set_fontsize(8)
            return s

        self.fig.text(0.60, 0.925, "WORLD", fontsize=8, color=C_MUTED)
        self.s_sea = slider(0.885, "sea level", 0.20, 0.70, self.sea_level)
        self.s_riv = slider(0.845, "river threshold", 40, 2000, self.river_thr, step=10, fmt="%.0f")

        self.fig.text(0.60, 0.795, "TIME & CLIMATE", fontsize=8, color=C_MUTED)
        self.s_speed = slider(0.755, "speed (days/s)", 0.02, 3.0, self.speed)
        self.s_dn = slider(0.715, "day/night depth", 0.0, 1.0, self.day_night)
        self.s_season = slider(0.675, "season amplitude", 0.0, 0.5, self.season_amp, fmt="%.3f")
        self.s_tide = slider(0.635, "tide amplitude", 0.0, 0.05, self.tide_amp, fmt="%.3f")

        for s, attr in [(self.s_sea, "sea_level"), (self.s_riv, "river_thr"),
                        (self.s_speed, "speed"), (self.s_dn, "day_night"),
                        (self.s_season, "season_amp"), (self.s_tide, "tide_amp")]:
            s.on_changed(lambda v, a=attr: (setattr(self, a, float(v)), self.refresh()))

        # ---- seed / transport ----
        def button(x, y, w, label):
            ax = self.fig.add_axes([x, y, w, 0.045])
            b = Button(ax, label, color=C_PANEL, hovercolor="#232b37")
            b.label.set_color(C_INK), b.label.set_fontsize(8.5)
            return b

        ax_seed = self.fig.add_axes([0.665, 0.545, 0.10, 0.045], facecolor=C_PANEL)
        self.tb_seed = TextBox(ax_seed, "seed  ", initial=str(self.seed),
                               color=C_PANEL, hovercolor="#232b37")
        self.tb_seed.label.set_color(C_INK), self.tb_seed.label.set_fontsize(8.5)
        self.tb_seed.text_disp.set_color(C_ACCENT)
        self.tb_seed.on_submit(self.on_seed)

        self.b_dice = button(0.775, 0.545, 0.05, "rnd")
        self.b_dice.on_clicked(self.on_dice)
        self.b_play = button(0.60, 0.475, 0.115, "pause")
        self.b_play.on_clicked(self.on_play)
        self.b_reset = button(0.725, 0.475, 0.115, "reset day 0")
        self.b_reset.on_clicked(lambda _e: (setattr(self, "t", 0.0), self.refresh()))

        # ---- export ----
        self.fig.text(0.60, 0.40, "EXPORT SEQUENCE", fontsize=8, color=C_MUTED)
        ax_fr = self.fig.add_axes([0.665, 0.335, 0.075, 0.042], facecolor=C_PANEL)
        self.tb_frames = TextBox(ax_fr, "frames  ", initial="96", color=C_PANEL, hovercolor="#232b37")
        ax_dt = self.fig.add_axes([0.83, 0.335, 0.075, 0.042], facecolor=C_PANEL)
        self.tb_dt = TextBox(ax_dt, "days/frame  ", initial="0.25", color=C_PANEL, hovercolor="#232b37")
        for tb in (self.tb_frames, self.tb_dt):
            tb.label.set_color(C_INK), tb.label.set_fontsize(8.5)
            tb.text_disp.set_color(C_ACCENT)

        ax_ex = self.fig.add_axes([0.60, 0.255, 0.305, 0.052])
        self.b_export = Button(ax_ex, "EXPORT PNG SEQUENCE", color=C_ACCENT, hovercolor="#f0b968")
        self.b_export.label.set_color("#16110a"), self.b_export.label.set_fontsize(9)
        self.b_export.on_clicked(self.on_export)
        self.export_note = self.fig.text(
            0.60, 0.20, "", fontsize=7.5, color=C_MUTED, va="top", wrap=True)

        self.fig.text(0.60, 0.09,
                      "click a thumbnail to change the main view\n"
                      f"1 day = 1 day/night cycle · year = {YEAR_DAYS:.0f} days · "
                      "seasons flip per hemisphere",
                      fontsize=7.5, color=C_MUTED, va="top")

        # animation timer
        self.timer = self.fig.canvas.new_timer(interval=50)
        self.timer.add_callback(self.tick)
        self.timer.start()
        self._last = time.time()
        self.refresh()

    # ---------------- events ----------------
    def _style_thumb_sel(self):
        for key, ax in self.thumb_axes.items():
            sel = key == self.layer
            self.thumb_titles[key].set_color(C_ACCENT if sel else C_MUTED)
            for sp in ax.spines.values():
                sp.set_edgecolor(C_ACCENT if sel else "#26303a")
                sp.set_linewidth(1.6 if sel else 0.8)

    def on_click(self, event):
        for key, ax in self.thumb_axes.items():
            if event.inaxes is ax:
                self.layer = key
                self._style_thumb_sel()
                self.refresh()
                return

    def on_seed(self, text):
        try:
            self.seed = int(text)
        except ValueError:
            return
        self.build_world()
        self.refresh()

    def on_dice(self, _e):
        self.seed = int(np.random.default_rng().integers(0, 100000))
        self.tb_seed.set_val(str(self.seed))   # triggers on_seed

    def on_play(self, _e):
        self.playing = not self.playing
        self.b_play.label.set_text("pause" if self.playing else "run")
        self.fig.canvas.draw_idle()

    # ---------------- frame ----------------
    def tick(self):
        now = time.time()
        dt = min(0.2, now - self._last)
        self._last = now
        if self.playing and not self.exporting:
            self.t += dt * self.speed
            self.refresh()

    def refresh(self):
        if self.exporting:
            return
        self.im_main.set_data(self.render(self.full, self.layer, self.t))
        for key, _ in LAYERS:
            self.thumb_ims[key].set_data(self.render(self.thumb, key, self.t))
        day, year = self.t % YEAR_DAYS, int(self.t // YEAR_DAYS)
        name = dict(LAYERS)[self.layer]
        self.title.set_text(
            f"seed {self.seed} · {self.size}² · {name.lower()} · day {day:.1f} · year {year}")
        self.export_note.set_text(
            f"exports the {name} layer at seed {self.seed}, {self.size}×{self.size},\n"
            "from the current sim time; meta.json included")
        self.fig.canvas.draw_idle()

    # ---------------- export ----------------
    def on_export(self, _e):
        try:
            frames = max(1, int(self.tb_frames.text))
            dpf = max(1e-3, float(self.tb_dt.text))
        except ValueError:
            print("export: bad frames / days-per-frame value")
            return
        self.export_sequence(frames, dpf)

    def export_sequence(self, frames, days_per_frame, out_root="exports"):
        self.exporting = True
        name = f"world_s{self.seed}_{self.layer}_{frames}f"
        out = Path(out_root) / name
        out.mkdir(parents=True, exist_ok=True)
        t0 = self.t
        print(f"exporting {frames} frames -> {out}/", flush=True)
        for f in range(frames):
            img = self.render(self.full, self.layer, t0 + f * days_per_frame)
            Image.fromarray(img).save(out / f"frame_{f:04d}.png")
            if f % 10 == 0 or f == frames - 1:
                self.b_export.label.set_text(f"EXPORTING… {f + 1}/{frames}")
                self.fig.canvas.draw_idle()
                plt.pause(0.001)
        meta = {
            "seed": self.seed, "size": self.size, "layer": self.layer,
            "frames": frames, "days_per_frame": days_per_frame, "start_day": t0,
            "sea_level": self.sea_level, "river_threshold": self.river_thr,
            "season_amplitude": self.season_amp, "tide_amplitude": self.tide_amp,
            "day_night_depth": self.day_night,
            "year_length_days": YEAR_DAYS, "tide_period_days": TIDE_PERIOD,
            "generator": "world-engine prototype M0-M2",
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        self.b_export.label.set_text("EXPORT PNG SEQUENCE")
        self.exporting = False
        self.refresh()
        print(f"  wrote {frames} frames + meta.json to {out}/")
        return out


def main():
    ap = argparse.ArgumentParser(description="World Engine layer console")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--size", type=int, default=256,
                    help="world resolution (256 = fluid, 512 = pretty)")
    args = ap.parse_args()
    console = Console(seed=args.seed, size=args.size)
    plt.show()
    return console


if __name__ == "__main__":
    main()
