"""
world_viewer.py — World Engine layer console (M0–M2 prototype), desktop UI.

All world math lives in world_core.py (shared with the web app in web/);
this file is only the matplotlib interface around it:

    - big map + 7 layer thumbnails, all animating; CLICK a thumbnail to
      promote that layer to the main view
    - sliders: sea level, river threshold, season amp, tide amp,
      day/night depth, sim speed
    - seed textbox + random button; play/pause; reset to day 0
    - EXPORT button: writes a PNG sequence of the current view layer at the
      current seed/settings to exports/, plus a meta.json of every parameter

Run:  python3 world_viewer.py [--seed 42] [--size 256]
      (512 looks better, 256 animates faster)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox

import world_core as wc
from world_core import LAYERS, YEAR_DAYS, TIDE_PERIOD

# console theme
C_BG, C_PANEL, C_INK, C_MUTED, C_ACCENT = "#0f1317", "#151b21", "#e8edf2", "#93a1af", "#e2a54f"


class Console:
    def __init__(self, seed=42, size=256):
        self.seed, self.size = seed, size
        self.t = 0.0
        self.playing = True
        self.layer = "composite"
        self.sea_level = 0.42
        self.river_thr = wc.default_river_threshold(size)
        self.season_amp, self.tide_amp = 0.18, 0.012
        self.day_night, self.speed = 0.65, 0.35
        self.exporting = False

        self.build_world()
        self.build_ui()

    # ---------------- world (heavy, on seed/size change only) ----------------
    def build_world(self):
        t0 = time.time()
        print(f"generating world seed={self.seed} size={self.size} ...", flush=True)
        self.full = wc.build_world(self.seed, self.size)
        self.thumb = self.full.strided(max(1, self.size // 96))
        print(f"  done in {time.time() - t0:.1f}s")

    def render(self, ws, layer, t):
        return wc.render(ws, layer, t, self.sea_level, self.river_thr,
                         self.season_amp, self.tide_amp, self.day_night)

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
