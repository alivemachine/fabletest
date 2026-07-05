"""
ecosim.py — EcoSim, the STATEFUL near-form substrate (M4). Everything else in
the render core is a pure, seekable function of t. This one is not: it
integrates forward and has MEMORY, so events leave lasting consequences. A
flood salinates the soil; a drought dries the forest; recovery is slow and
only spreads inward from neighbouring cells, so an isolated barren zone stays
barren until life reaches it again — which is how a world can fail to recover.
Driven live by the sliders (sea level, seasons); reset() returns it to the
pristine world. It is deliberately NOT scrubbable backward (that is what
"consequences" means) — it only runs forward or resets. Coarse (HIST_SIZE)
and cheap.
"""

import numpy as np

from common import smoothstep, _neigh4
from history import HIST_SIZE, _window_indices


class EcoSim:
    def __init__(self, ws, seed=0):
        H = HIST_SIZE
        self.H = H
        self.e = ws._coarse(ws.elev).astype(np.float32)
        self.m = ws._coarse(ws.moist).astype(np.float32)
        yn = (np.arange(H, dtype=np.float32) / H)[:, None]
        self.lat = np.repeat(1 - np.abs(yn - 0.5) * 2, H, axis=1).astype(np.float32)
        self.lat_signed = np.repeat((0.5 - yn) * 2, H, axis=1).astype(np.float32)
        self.sea0 = 0.42
        self.reset()

    def _climate(self, season_off):
        temp = np.clip(self.lat + season_off * self.lat_signed
                       - np.clip(self.e - self.sea_ref, 0, 1) * 0.9, 0, 1)
        warmth = np.clip((temp - 0.26) / 0.55, 0, 1)
        wet = smoothstep((self.m - 0.1) / 0.8)
        return warmth, wet

    def reset(self):
        """Back to the pristine, climax-state world (day 0): full soil, and
        vegetation/fauna at the climate's potential, so health == 1 everywhere
        and the flora/fauna layers look exactly like their pure baseline until
        something happens to them."""
        self.sea_ref = float(self.sea0)
        self.t = 0.0
        land = (self.e >= self.sea0).astype(np.float32)
        warmth, wet = self._climate(0.0)
        clim = warmth * (0.30 + 0.70 * wet) * land          # climatic potential
        self.clim = clim.astype(np.float32)
        self.fert = land.copy()                             # climax soil = 1 on land
        self.veg = clim.astype(np.float32)                  # at full potential
        self.fauna = (0.6 * clim).astype(np.float32)
        self.civ = ((clim > 0.55) * 0.35).astype(np.float32)   # some seed settlements
        self.scorch = np.zeros((self.H, self.H), np.float32)

    def step(self, dt_days, sea_level, season_off):
        """Advance the ecosystem by dt_days under the current sliders."""
        if dt_days <= 0:
            return
        n = int(min(48, max(1, np.ceil(dt_days / 4.0))))    # sub-step for stability
        h = min(dt_days / n, 6.0)
        for _ in range(n):
            self._micro(h, float(sea_level), float(season_off))
        self.t += dt_days

    def _micro(self, h, sea_level, season_off):
        e = self.e
        warmth, wet = self._climate(season_off)
        under = e < sea_level
        land = ~under
        submerged = under & (e >= self.sea_ref)        # land the sea just covered
        clim = warmth * (0.30 + 0.70 * wet) * land       # climatic potential
        self.clim = clim.astype(np.float32)              # (for health readout)
        cap = clim * self.fert * (1 - self.scorch)
        dry = np.clip(warmth - 0.75 * wet - 0.05, 0, 1)  # hot & dry -> fire/desert

        veg, fauna, civ, fert, scorch = (self.veg, self.fauna, self.civ,
                                         self.fert, self.scorch)

        # --- flood: submerged biota declines, soil salinates ---
        veg = np.where(under, veg * np.exp(-0.6 * h), veg)
        fauna = np.where(under, fauna * np.exp(-0.5 * h), fauna)
        civ = np.where(under, civ * np.exp(-0.7 * h), civ)
        fert = np.where(submerged, fert - 0.008 * h, fert)
        scorch = np.where(submerged, scorch + 0.020 * h, scorch)

        # --- fire / drought: needs both heat-dryness AND fuel (vegetation), so
        #     it strikes grass/savanna in hot summers, not bare desert ---
        burn = np.clip(dry - 0.40, 0, 1) * veg * land
        ignite = burn * (burn > 0.03)
        veg = veg - 0.70 * ignite * h
        fauna = fauna - 0.45 * ignite * h
        scorch = scorch + 0.40 * ignite * h
        desert = land & (dry > 0.45) & (veg < 0.10)     # bare hot ground erodes
        fert = np.where(desert, fert - 0.004 * h, fert)

        # --- recovery on dry land: growth needs a seed (self or neighbour), so
        #     cleared, isolated cells cannot restart until life spreads back in ---
        cap_pos = cap > 0.02
        vseed = 0.015 + 0.85 * veg + 0.5 * _neigh4(veg)
        veg = np.where(land, veg + 0.045 * h * vseed
                       * np.clip(1 - veg / (cap + 1e-3), 0, 1) * cap_pos, veg)
        fcap = 0.9 * veg
        fseed = 0.02 + 0.85 * fauna + 0.5 * _neigh4(fauna)
        fauna = np.where(land, fauna + 0.06 * h * fseed
                         * np.clip(1 - fauna / (fcap + 1e-3), 0, 1) * (fcap > 0.02), fauna)
        # slow soil rebuild (needs life nearby) and scar fade — the "much time
        # must pass" knobs, on a timescale of years not days
        fert = fert + h * (0.0016 * veg + 0.0009 * _neigh4(veg)) * (1 - fert) * land
        scorch = scorch - 0.0025 * h

        # --- civilization: grows on food, collapses without it, recolonises
        #     only from surviving neighbours ---
        food = 0.5 * veg + 0.5 * fauna
        cseed = 0.55 * civ + 0.35 * _neigh4(civ)
        ok = land & (food > 0.28)
        civ = civ + np.where(ok, 0.02 * h * (0.04 + cseed) * (1 - civ), 0)
        civ = civ - np.where(~ok, 0.05 * h * civ, 0)
        civ = np.where(food < 0.14, civ * np.exp(-0.12 * h), civ)  # food-collapse decline
        fauna = fauna - 0.03 * h * civ * fauna          # hunting pressure
        veg = veg - 0.01 * h * civ * veg                # land clearing

        self.veg = np.clip(veg, 0, 1).astype(np.float32)
        self.fauna = np.clip(fauna, 0, 1).astype(np.float32)
        self.civ = np.clip(civ, 0, 1).astype(np.float32)
        self.fert = np.clip(fert, 0.02, 1).astype(np.float32)
        self.scorch = np.clip(scorch, 0, 1).astype(np.float32)
        # the coastline the ecosystem is adapted to drifts toward the imposed
        # level (slowly), so a held sea level becomes the new normal
        self.sea_ref += 0.02 * h * (sea_level - self.sea_ref)

    def sample(self, ws):
        """Upsample the coarse state to the render window (honours pan/zoom)."""
        up = _window_indices(ws)
        return {k: getattr(self, k)[up] for k in
                ("veg", "fauna", "fert", "civ", "scorch", "clim")}
