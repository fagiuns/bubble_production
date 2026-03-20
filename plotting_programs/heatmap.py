import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from bubble_tools import E_constv


# Grid

R = 1e-3
Z_min = 10**-12 * R**-4
Z_max = R**-4
m_min = - np.log10(R)
m_max = 9 - np.log10(R)
gamma_max = np.log10(10**15 * R)
m_vals = np.logspace(m_min, m_max, 300)
gamma_vals = np.logspace(0, gamma_max, 300)

M, G = np.meshgrid(m_vals, gamma_vals)
J_0 = 1/(10**16 * R**4)

Z = E_constv(1e-9 * M, R * G, G, J_0)


# Main heatmap only in the wanted range

Z_main = np.ma.masked_where((Z < Z_min) | (Z > Z_max) | (Z <= 0), Z)

cmap = plt.get_cmap("viridis").copy()
cmap.set_bad("white")

fig, ax = plt.subplots(figsize=(8, 6))

pcm = ax.pcolormesh(
    M,
    G,
    Z_main,
    shading="auto",
    cmap=cmap,
    norm=LogNorm(vmin=Z_min, vmax=Z_max)
)

cbar = plt.colorbar(pcm, ax=ax)
cbar.set_label(r"$E$ (GeV)")


# Hatched region: Z > Z_max  -> overshoot

if np.nanmax(Z) > Z_max:
    cf_over = ax.contourf(
        M, G, Z,
        levels=[Z_max, np.nanmax(Z)],
        colors="none",
        hatches=["\\\\\\"]
    )
    cf_over.set_edgecolor("orange")


# Hatched region: 0 < Z < Z_min  -> negligible

Z_pos = np.where(Z > 0, Z, np.nan)
min_pos = np.nanmin(Z_pos)

cf_under = ax.contourf(
    M, G, Z_pos,
    levels=[np.nanmin(Z_pos), Z_min],
    colors="none",
    hatches=["\\\\\\"]
)

cf_under.set_edgecolor("blue")

# Axes

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$m$ (eV)")
ax.set_ylabel(r"$\gamma$")
ax.set_title(rf"$\Lambda = 10^{{{-np.log10(R):.0f}}}$ GeV")


# Labels for hatched regions
# Choose positions that you can tune if needed
# Labels fixed relative to the plot window, not the data
ax.text(
    0.03, 0.92, "overproduced",
    transform=ax.transAxes,
    color="orange",
    fontsize=10,
    ha="left",
    va="top",
    zorder=10,
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5)
)

ax.text(
    0.97, 0.05, "negligible",
    transform=ax.transAxes,
    color="blue",
    fontsize=10,
    ha="right",
    va="bottom",
    zorder=10,
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5)
)


# Relativistic / non-relativistic line

T_bbn = 1e-3   # BBN temperature

gamma_line = 1/(T_bbn * R)

ax.axhline(gamma_line, color="red", linewidth=2, zorder=5)

# place labels near the left side
xmin, xmax = ax.get_xlim()
x_text = xmin * 2

ax.text(
    x_text,
    gamma_line * 1.4,
    "relativistic",
    color="red",
    fontsize=9,
    ha="left",
    va="bottom",
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5)
)

ax.text(
    x_text,
    gamma_line / 1.4,
    "non relativistic",
    color="red",
    fontsize=9,
    ha="left",
    va="top",
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5)
)

plt.tight_layout()
plt.show()
