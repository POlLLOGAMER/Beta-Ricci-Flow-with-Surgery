"""
Ricci Flow with Surgery (axisymmetric 3-manifold model) in one Colab-friendly cell.

This is a practical numerical implementation of Hamilton–DeTurck Ricci flow with
"surgery" for rotationally symmetric metrics

    g = ds^2 + f(s,t)^2 d\Omega^2,

where d\Omega^2 is the unit round metric on S^2.

The evolution equation used is the symmetry-reduced Ricci flow in arclength gauge:

    \partial_t f = f_{ss} - (1 - f_s^2)/f,

with Neumann conditions f_s(0)=f_s(L)=0.  A "surgery" is triggered when a neck radius
falls below a threshold and has high curvature; the neck region is excised and capped
with smooth spherical caps, then the flow continues.

NOTE:
- This is a real PDE + surgery code for the axisymmetric setting (not a full general
  3-manifold triangulation engine).
- It is designed for educational / experimental use in a single Google Colab cell.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class SurgeryConfig:
    neck_radius_trigger: float = 0.16
    neck_curv_trigger: float = 40.0
    surgery_half_width: float = 0.17
    cap_radius: float = 0.23
    smooth_sigma: float = 2.0


@dataclass
class FlowConfig:
    n: int = 801
    L: float = 2.8
    dt: float = 2.0e-6
    t_max: float = 3.0e-2
    snapshot_every: int = 350
    max_snapshots: int = 240


class AxisymmetricRicciFlowWithSurgery:
    def __init__(self, flow_cfg: FlowConfig, surg_cfg: SurgeryConfig):
        self.cfg = flow_cfg
        self.surg = surg_cfg
        self.s = np.linspace(0.0, self.cfg.L, self.cfg.n)
        self.ds = self.s[1] - self.s[0]
        self.t = 0.0
        self.surgery_count = 0

        # Initial dumbbell profile on S^3-like topology:
        # two bulges joined by a thin neck near center.
        x = (self.s - self.cfg.L / 2) / (self.cfg.L / 2)
        self.f = (
            0.31
            + 0.45 * (1.0 - x**2)
            - 0.25 * np.exp(-(x / 0.17) ** 2)
            + 0.04 * np.cos(2 * np.pi * x)
        )
        self.f = np.clip(self.f, 0.08, None)

        self.times = [self.t]
        self.snapshots = [self.f.copy()]
        self.events = []

    def d1(self, u: np.ndarray) -> np.ndarray:
        out = np.zeros_like(u)
        out[1:-1] = (u[2:] - u[:-2]) / (2 * self.ds)
        # Neumann: u_s = 0 at boundaries
        out[0] = 0.0
        out[-1] = 0.0
        return out

    def d2(self, u: np.ndarray) -> np.ndarray:
        out = np.zeros_like(u)
        out[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / (self.ds**2)
        # Ghost reflection for Neumann BC
        out[0] = 2 * (u[1] - u[0]) / (self.ds**2)
        out[-1] = 2 * (u[-2] - u[-1]) / (self.ds**2)
        return out

    def scalar_curvature(self, f: np.ndarray) -> np.ndarray:
        # For g = ds^2 + f^2 dΩ^2 in 3D:
        # R = -4 f_ss/f + 2(1 - f_s^2)/f^2
        fs = self.d1(f)
        fss = self.d2(f)
        return -4.0 * fss / np.maximum(f, 1e-8) + 2.0 * (1.0 - fs**2) / np.maximum(f, 1e-8) ** 2

    def smooth(self, u: np.ndarray, sigma: float) -> np.ndarray:
        if sigma <= 0:
            return u
        radius = max(1, int(4 * sigma))
        x = np.arange(-radius, radius + 1)
        k = np.exp(-(x**2) / (2 * sigma**2))
        k /= k.sum()
        padded = np.pad(u, radius, mode="edge")
        return np.convolve(padded, k, mode="valid")

    def rhs(self, f: np.ndarray) -> np.ndarray:
        fs = self.d1(f)
        fss = self.d2(f)
        return fss - (1.0 - fs**2) / np.maximum(f, 1e-8)

    def step(self):
        # Heun RK2 for robustness
        k1 = self.rhs(self.f)
        f1 = np.clip(self.f + self.cfg.dt * k1, 0.02, None)
        k2 = self.rhs(f1)
        self.f = np.clip(self.f + 0.5 * self.cfg.dt * (k1 + k2), 0.02, None)
        self.t += self.cfg.dt

    def detect_surgery_neck(self):
        f = self.f
        R = self.scalar_curvature(f)
        i = np.argmin(f)
        neck_r = f[i]
        neck_R = R[i]

        # avoid boundary artifacts
        if i < 8 or i > len(f) - 9:
            return None

        if neck_r < self.surg.neck_radius_trigger and neck_R > self.surg.neck_curv_trigger:
            return i, neck_r, neck_R
        return None

    def perform_surgery(self, i_neck: int):
        s = self.s
        f = self.f.copy()
        s0 = s[i_neck]
        hw = self.surg.surgery_half_width

        left_mask = s < (s0 - hw)
        right_mask = s > (s0 + hw)

        if left_mask.sum() < 16 or right_mask.sum() < 16:
            return False

        s_left = s[left_mask]
        s_right = s[right_mask]
        f_left = f[left_mask]
        f_right = f[right_mask]

        # Create smooth caps near cut boundaries using spherical-cap model
        def cap_segment(s_seg, f_seg, at_left_end: bool):
            g = f_seg.copy()
            m = min(50, len(g) // 2)
            if m < 6:
                return g

            r_cap = self.surg.cap_radius
            if at_left_end:
                anchor = g[-m - 1]
                j = np.arange(m)
                u = (m - j) / m
                cap = np.sqrt(np.maximum(r_cap**2 - (r_cap * (1 - u)) ** 2, 0.0))
                cap = cap / max(cap.max(), 1e-8) * anchor
                g[-m:] = cap
            else:
                anchor = g[m]
                j = np.arange(m)
                u = j / m
                cap = np.sqrt(np.maximum(r_cap**2 - (r_cap * (1 - u)) ** 2, 0.0))
                cap = cap / max(cap.max(), 1e-8) * anchor
                g[:m] = cap

            return g

        f_left_capped = cap_segment(s_left, f_left, at_left_end=True)
        f_right_capped = cap_segment(s_right, f_right, at_left_end=False)

        # Build a connected manifold by selecting the larger side after surgery
        # (mimics discarding a singular neck component in one branch).
        if f_left_capped.mean() >= f_right_capped.mean():
            s_new = np.linspace(0.0, self.cfg.L, len(f_left_capped))
            x_old = (s_left - s_left.min()) / (s_left.max() - s_left.min())
            x_new = (s_new - s_new.min()) / (s_new.max() - s_new.min())
            f_new = np.interp(x_new, x_old, f_left_capped)
        else:
            s_new = np.linspace(0.0, self.cfg.L, len(f_right_capped))
            x_old = (s_right - s_right.min()) / (s_right.max() - s_right.min())
            x_new = (s_new - s_new.min()) / (s_new.max() - s_new.min())
            f_new = np.interp(x_new, x_old, f_right_capped)

        # Resample back to original grid and smooth surgery seam
        self.f = np.interp(self.s, s_new, f_new)
        self.f = self.smooth(np.clip(self.f, 0.04, None), self.surg.smooth_sigma)
        self.surgery_count += 1
        self.events.append((self.t, "surgery", i_neck))
        return True

    def run(self):
        nsteps = int(self.cfg.t_max / self.cfg.dt)
        for k in range(nsteps):
            self.step()
            neck = self.detect_surgery_neck()
            if neck is not None:
                i, nr, nR = neck
                ok = self.perform_surgery(i)
                if ok:
                    print(f"[t={self.t:.6f}] SURGERY #{self.surgery_count} at s≈{self.s[i]:.3f}, neck={nr:.4f}, R={nR:.2f}")

            if (k + 1) % self.cfg.snapshot_every == 0:
                self.times.append(self.t)
                self.snapshots.append(self.f.copy())
                if len(self.snapshots) >= self.cfg.max_snapshots:
                    break

        # include final
        self.times.append(self.t)
        self.snapshots.append(self.f.copy())


def run_demo():
    flow_cfg = FlowConfig(
        n=801,
        L=2.8,
        dt=2.0e-6,
        t_max=3.0e-2,
        snapshot_every=350,
        max_snapshots=240,
    )
    surg_cfg = SurgeryConfig(
        neck_radius_trigger=0.16,
        neck_curv_trigger=40.0,
        surgery_half_width=0.17,
        cap_radius=0.23,
        smooth_sigma=2.0,
    )

    sim = AxisymmetricRicciFlowWithSurgery(flow_cfg, surg_cfg)
    sim.run()

    # Plot evolution profiles
    plt.figure(figsize=(10, 6))
    ncurves = len(sim.snapshots)
    pick = np.linspace(0, ncurves - 1, min(16, ncurves), dtype=int)
    cmap = plt.cm.plasma(np.linspace(0.15, 0.95, len(pick)))
    for c, j in zip(cmap, pick):
        plt.plot(sim.s, sim.snapshots[j], color=c, alpha=0.9, lw=1.6)

    plt.title(
        f"Axisymmetric Ricci Flow with Surgery | surgeries={sim.surgery_count} | t_final={sim.t:.5f}",
        fontsize=11,
    )
    plt.xlabel("s")
    plt.ylabel("radius f(s,t)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

    # Curvature at final time
    Rf = sim.scalar_curvature(sim.f)
    plt.figure(figsize=(10, 4))
    plt.plot(sim.s, Rf, "k", lw=1.5)
    plt.title("Final scalar curvature R(s)")
    plt.xlabel("s")
    plt.ylabel("R")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()

    print(f"Done. Total surgeries performed: {sim.surgery_count}")


if __name__ == "__main__":
    run_demo()
