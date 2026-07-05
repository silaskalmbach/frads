"""Transient lumped-capacitance thermal model of a Complex Fenestration System.

Standard EnergyPlus window / Construction:ComplexFenestrationState heat balances
are quasi-steady (massless glass layers). For strongly *absorbing* glazings
(e.g. electrochromic units in the tinted state) the panes store the absorbed
solar and release it inward with a time lag of ~1-2 h, which the massless model
misses. This module adds a generic 1-D RC network — one capacitive node per
glass layer — that, given the per-layer solar absorptance (from the aBSDF) and
the panes' physical/thermal properties, computes the *transient* inward heat
gain and the *steady* (massless) reference, and returns their difference as an
energy-neutral "lag correction".

The model is intentionally generic: it builds itself from any
``frads.window.GlazingSystem`` (any number of layers/gaps, any tint state) plus
config-supplied density / specific heat (which the aBSDF does not carry). It is
pure and deterministic — no EnergyPlus dependency — so it can be unit-tested and
validated offline against measured pane temperatures.

Coupling to a building energy simulation (see frads-gym):
  * the TRANSMITTED solar stays in the optical CFS model (prompt, switch-
    dependent, onto the floor) — untouched here;
  * only the ABSORBED -> inward path is given thermal mass. Inject
    ``lag_correction_wm2 * glazed_area`` as a zone OtherEquipment power level.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

SIGMA = 5.670374419e-8  # Stefan-Boltzmann constant [W/(m2 K4)]

# ISO 15099 gas conductivity   k(T) = A + B*T   [W/(m K)], T in Kelvin.
_GAS_K = {
    "air": (2.873e-3, 7.760e-5),
    "argon": (2.285e-3, 5.149e-5),
    "krypton": (9.443e-4, 2.826e-5),
    "xenon": (4.538e-4, 1.723e-5),
}


def _klems_omegas() -> list[float]:
    """Klems-full projected solid angles (cosine-weighted, sum = pi)."""
    from frads.ep2rad import OMEGAS

    return OMEGAS["kf"]


def klems_hemispherical_average(vec: list[float], omegas: list[float] | None = None) -> float:
    """Cosine-weighted (projected-solid-angle) hemispherical average of a
    Klems-full directional vector (e.g. one layer's 145 absorptance values).
    Falls back to a plain mean if the length does not match the Klems basis."""
    if isinstance(vec, (int, float)):
        return float(vec)
    if omegas is None:
        omegas = _klems_omegas()
    if len(vec) != len(omegas):
        return sum(vec) / len(vec)
    return sum(v * o for v, o in zip(vec, omegas)) / sum(omegas)


@dataclass
class PaneThermalProps:
    """Thermal/optical properties of one solid glazing layer (per unit area)."""

    thickness_m: float
    conductivity: float
    emissivity_front: float
    emissivity_back: float
    ir_transmittance: float = 0.0
    density: float = 2500.0          # kg/m3   (NOT in the aBSDF -> config/default)
    specific_heat: float = 840.0     # J/(kg K)(NOT in the aBSDF -> config/default)

    @property
    def capacitance(self) -> float:
        """Areal heat capacity C = rho * cp * d  [J/(m2 K)]."""
        return self.density * self.specific_heat * self.thickness_m

    @property
    def half_resistance(self) -> float:
        """Conduction resistance of half the pane  [m2 K/W]."""
        k = self.conductivity if self.conductivity and self.conductivity > 0 else 1.0
        return 0.5 * self.thickness_m / k


@dataclass
class GapThermalProps:
    """Gas gap between two panes (conduction + optional convection via Nu)."""

    thickness_m: float
    gas_mix: list[tuple[str, float]]   # [(gas_name, ratio), ...]

    def gas_conductivity(self, t_mean_k: float) -> float:
        k = 0.0
        for name, ratio in self.gas_mix:
            a, b = _GAS_K.get(name.lower(), _GAS_K["air"])
            k += ratio * (a + b * t_mean_k)
        return k

    def conductance(self, t_mean_k: float, nu: float = 1.0) -> float:
        """Conductive(/convective) conductance  [W/(m2 K)].  nu=1 -> pure conduction."""
        return nu * self.gas_conductivity(t_mean_k) / self.thickness_m


@dataclass
class StepResult:
    pane_temps_c: list[float]
    inward_transient_wm2: float
    inward_steady_wm2: float
    lag_correction_wm2: float


def _thomas(a: list[float], b: list[float], c: list[float], d: list[float]) -> list[float]:
    """Solve a tridiagonal system (sub a, diag b, super c, rhs d). a[0], c[-1] unused."""
    n = len(b)
    cp = [0.0] * n
    dp = [0.0] * n
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = [0.0] * n
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


class CFSThermalModel:
    """Lumped-capacitance RC model for one glazing system / window.

    Nodes 0..N-1 sit at the mid-plane of each glass layer (outside -> inside).
    State ``self.T`` (Kelvin) persists across calls and across tint switches.
    """

    def __init__(
        self,
        panes: list[PaneThermalProps],
        gaps: list[GapThermalProps],
        absorptance_front: list,          # per-pane: scalar or Klems-145 vector
        *,
        absorptance_mode: str = "hemispherical",
        h_out: float = 23.0,              # outdoor film (conv+rad), W/(m2 K)
        h_in: float = 8.0,                # indoor film (conv), W/(m2 K)
        gap_nu: float = 1.0,
        include_gap_radiation: bool = True,
        include_inside_radiation: bool = True,
        init_temp_c: float = 20.0,
    ):
        if len(gaps) != len(panes) - 1:
            raise ValueError(f"expected {len(panes) - 1} gaps, got {len(gaps)}")
        if absorptance_mode != "hemispherical":
            raise NotImplementedError(
                f"absorptance_mode={absorptance_mode!r} is not implemented; "
                "only 'hemispherical' is supported. The 'angular' refinement "
                "(map incidence cosine -> Klems polar band for the beam "
                "fraction) was never wired up; previously it silently fell "
                "back to the hemispherical average, yielding wrong absorptance."
            )
        self.panes = panes
        self.gaps = gaps
        self.absorptance_mode = absorptance_mode
        self.h_out = h_out
        self.h_in = h_in
        self.gap_nu = gap_nu
        self.include_gap_radiation = include_gap_radiation
        self.include_inside_radiation = include_inside_radiation
        self._omegas = _klems_omegas()
        self.set_absorptance(absorptance_front)
        self.T = [init_temp_c + 273.15] * len(panes)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_glazing_system(cls, gs, *, overrides: dict | None = None, **kw) -> "CFSThermalModel":
        """Build from a frads GlazingSystem. ``overrides`` maps layer index
        (int or str) -> {"thickness_m":..., "density":..., "specific_heat":...,
        "conductivity":...}. ``thickness_m`` overrides the layer's *thermal*
        thickness (e.g. when the aBSDF geometry mis-distributes the panes vs the
        real build-up) while the per-layer solar absorptance is kept from the
        aBSDF."""
        overrides = overrides or {}
        panes = []
        for i, layer in enumerate(gs.layers):
            o = overrides.get(i, overrides.get(str(i), {}))
            panes.append(
                PaneThermalProps(
                    thickness_m=o.get("thickness_m", layer.thickness_m),
                    conductivity=o.get("conductivity", layer.conductivity or 1.0),
                    emissivity_front=layer.emissivity_front,
                    emissivity_back=layer.emissivity_back,
                    ir_transmittance=layer.ir_transmittance,
                    density=o.get("density", 2500.0),
                    specific_heat=o.get("specific_heat", 840.0),
                )
            )
        gaps = [
            GapThermalProps(thickness_m=g.thickness_m, gas_mix=[(x.gas, x.ratio) for x in g.gas])
            for g in gs.gaps
        ]
        return cls(panes, gaps, gs.solar_front_absorptance, **kw)

    def set_absorptance(self, absorptance_front: list) -> None:
        """Swap absorptance (e.g. on a tint switch) while keeping the state T."""
        self._abs_raw = absorptance_front
        self._alpha_hemis = [
            klems_hemispherical_average(a, self._omegas) for a in absorptance_front
        ]

    def alpha_for_step(self, incidence_cos: float | None = None) -> list[float]:
        # Only the hemispherical average is implemented (enforced in __init__).
        # An angular mode (incidence_cos -> Klems polar band for the beam
        # fraction) would go here; ``incidence_cos`` is accepted for that future
        # signature but is currently unused.
        return self._alpha_hemis

    # -- core --------------------------------------------------------------
    def _conductances(self, T: list[float], h_in: float):
        """Inter-node + boundary conductances [W/(m2 K)] at current temps T."""
        n = len(self.panes)
        g_inter = []
        for j, gap in enumerate(self.gaps):
            t_mean = 0.5 * (T[j] + T[j + 1])
            g_gap = gap.conductance(t_mean, self.gap_nu)
            if self.include_gap_radiation:
                eps_a, eps_b = self.panes[j].emissivity_back, self.panes[j + 1].emissivity_front
                denom = 1.0 / eps_a + 1.0 / eps_b - 1.0
                if denom > 0:
                    g_gap += 4.0 * SIGMA * t_mean ** 3 / denom
            r = self.panes[j].half_resistance + 1.0 / g_gap + self.panes[j + 1].half_resistance
            g_inter.append(1.0 / r)
        # outdoor boundary (node 0)
        g_out = 1.0 / (1.0 / self.h_out + self.panes[0].half_resistance)
        # indoor boundary (node N-1): convection (+ linearized radiation to room)
        h_i = h_in
        if self.include_inside_radiation:
            h_i += 4.0 * SIGMA * self.panes[-1].emissivity_back * T[-1] ** 3
        g_in = 1.0 / (1.0 / h_i + self.panes[-1].half_resistance)
        return g_inter, g_out, g_in

    def _assemble(self, T, g_inter, g_out, g_in, alpha, incident, t_out_k, t_in_k, cap_over_dt):
        """Build tridiagonal (a,b,c,d). cap_over_dt[i] = C_i/dt (0 for steady)."""
        n = len(self.panes)
        a = [0.0] * n
        b = [0.0] * n
        c = [0.0] * n
        d = [0.0] * n
        for i in range(n):
            g_left = g_out if i == 0 else g_inter[i - 1]
            g_right = g_in if i == n - 1 else g_inter[i]
            b[i] = cap_over_dt[i] + g_left + g_right
            if i > 0:
                a[i] = -g_inter[i - 1]
            if i < n - 1:
                c[i] = -g_inter[i]
            d[i] = cap_over_dt[i] * T[i] + alpha[i] * incident
            if i == 0:
                d[i] += g_out * t_out_k
            if i == n - 1:
                d[i] += g_in * t_in_k
        return a, b, c, d

    def steady_inward(self, incident, t_out_c, t_in_c, *, h_in=None, incidence_cos=None) -> float:
        """Massless (C=0) inward gain [W/m2] — the EnergyPlus-CFS reference."""
        h_in = self.h_in if h_in is None else h_in
        alpha = self.alpha_for_step(incidence_cos)
        t_out_k, t_in_k = t_out_c + 273.15, t_in_c + 273.15
        T = list(self.T)
        for _ in range(3):  # few fixed-point iterations (G depends on T)
            g_inter, g_out, g_in = self._conductances(T, h_in)
            a, b, c, d = self._assemble(T, g_inter, g_out, g_in, alpha, incident,
                                        t_out_k, t_in_k, [0.0] * len(self.panes))
            T = _thomas(a, b, c, d)
        _, _, g_in = self._conductances(T, h_in)
        return g_in * (T[-1] - t_in_k)

    def step(self, incident, t_out_c, t_in_c, dt_s, *, h_in=None, incidence_cos=None,
             max_sub_dt=120.0) -> StepResult:
        """Advance the transient state by dt_s (with sub-stepping) and return the
        inward gains. Inputs held constant over dt_s."""
        h_in = self.h_in if h_in is None else h_in
        alpha = self.alpha_for_step(incidence_cos)
        t_out_k, t_in_k = t_out_c + 273.15, t_in_c + 273.15
        n_sub = max(1, int(math.ceil(dt_s / max_sub_dt)))
        sub_dt = dt_s / n_sub
        caps = [p.capacitance for p in self.panes]
        for _ in range(n_sub):
            g_inter, g_out, g_in = self._conductances(self.T, h_in)
            cap_over_dt = [c / sub_dt for c in caps]
            a, b, c, d = self._assemble(self.T, g_inter, g_out, g_in, alpha, incident,
                                        t_out_k, t_in_k, cap_over_dt)
            self.T = _thomas(a, b, c, d)
        _, _, g_in = self._conductances(self.T, h_in)
        inward_transient = g_in * (self.T[-1] - t_in_k)
        inward_steady = self.steady_inward(incident, t_out_c, t_in_c, h_in=h_in,
                                           incidence_cos=incidence_cos)
        return StepResult(
            pane_temps_c=[t - 273.15 for t in self.T],
            inward_transient_wm2=inward_transient,
            inward_steady_wm2=inward_steady,
            lag_correction_wm2=inward_transient - inward_steady,
        )

    @property
    def pane_temps_c(self) -> list[float]:
        return [t - 273.15 for t in self.T]
