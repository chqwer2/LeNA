import numpy as np
import matplotlib.pyplot as plt

# ---- Spline activation (shape-preserving cubic Hermite / PCHIP-like) ----
def _pchip_slopes(xk, yk):
    xk = np.asarray(xk, dtype=float)
    yk = np.asarray(yk, dtype=float)

    h = np.diff(xk)
    delta = np.diff(yk) / h
    m = np.zeros_like(yk)

    # Interior slopes
    for i in range(1, len(yk) - 1):
        if delta[i - 1] == 0 or delta[i] == 0 or np.sign(delta[i - 1]) != np.sign(delta[i]):
            m[i] = 0.0
        else:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])

    # Endpoint slopes
    m[0] = ((2 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1]) if len(h) > 1 else delta[0]
    m[-1] = ((2 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2]) if len(h) > 1 else delta[-1]

    # Limit endpoints to preserve shape
    if np.sign(m[0]) != np.sign(delta[0]):
        m[0] = 0.0
    elif abs(m[0]) > 3 * abs(delta[0]):
        m[0] = 3 * delta[0]

    if np.sign(m[-1]) != np.sign(delta[-1]):
        m[-1] = 0.0
    elif abs(m[-1]) > 3 * abs(delta[-1]):
        m[-1] = 3 * delta[-1]

    return m

def spline_activation(x, xk, yk):
    x = np.asarray(x, dtype=float)
    xk = np.asarray(xk, dtype=float)
    yk = np.asarray(yk, dtype=float)

    if np.any(np.diff(xk) <= 0):
        raise ValueError("xk must be strictly increasing.")

    m = _pchip_slopes(xk, yk)

    # Clamp outside knot range
    y = np.empty_like(x)
    y[x <= xk[0]] = yk[0]
    y[x >= xk[-1]] = yk[-1]

    mask = (x > xk[0]) & (x < xk[-1])
    xi = x[mask]
    idx = np.searchsorted(xk, xi) - 1
    idx = np.clip(idx, 0, len(xk) - 2)

    x0 = xk[idx]
    x1 = xk[idx + 1]
    y0 = yk[idx]
    y1 = yk[idx + 1]
    m0 = m[idx]
    m1 = m[idx + 1]

    h = x1 - x0
    t = (xi - x0) / h

    h00 = (2 * t**3 - 3 * t**2 + 1)
    h10 = (t**3 - 2 * t**2 + t)
    h01 = (-2 * t**3 + 3 * t**2)
    h11 = (t**3 - t**2)

    y[mask] = h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1
    return y


if __name__ == "__main__":
    x = np.linspace(-5, 5, 800)

    # Shared knot x-positions
    # Shared knot x-positions
    xk_soft = np.array([-5, -2, 0, 2, 5], dtype=float)
    yk_soft = np.array([-2.8, -0.9, 0.0, 1.6, 3.6])

    # -------------------------
    # Bumpy spline (K = 9)
    # high capacity, non-monotone
    # -------------------------
    xk_bumpy = np.array([-5, -3.5, -2.2, -1.2, 0, 1.2, 2.2, 3.5, 5], dtype=float)
    yk_bumpy = np.array([
        -2.2,  # reduced negative tail
        -1.8,  # smoother rise
        0.4,  # smaller positive bump
        -0.3,  # gentler dip
        0.0,  # anchor at zero
        1.5,  # reduced post-zero peak
        0.8,  # mild dip (still non-monotone)
        2.6,  # smoother recovery
        2.0  # lower final amplitude
    ])

    # -------------------------
    # Saturating spline (K = 7)
    # strong saturation on + side
    # -------------------------
    xk_sat = np.array([-5, -3, -1.5, 0, 1.5, 3, 5], dtype=float)

    yk_sat = np.array([
        -3.8,  # stronger negative saturation
        -2.9,  # steeper rise from negative tail
        -1.0,  # sharp curvature before zero
        0.0,  # anchor at origin
        1.2,  # strong post-zero activation
        1.35,  # start of saturation
        1.40  # flat positive tail (saturation)
    ])

    y_soft = spline_activation(x, xk_soft, yk_soft)
    y_bump = spline_activation(x, xk_bumpy, yk_bumpy)
    y_sat = spline_activation(x, xk_sat, yk_sat)

    # Figure (your style)
    fig, ax = plt.subplots(figsize=(6, 5))

    limit = 5
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    ax.annotate("", xy=(limit, 0), xytext=(-limit, 0),
                arrowprops=dict(arrowstyle="->", linewidth=1))
    ax.annotate("", xy=(0, limit), xytext=(0, -limit),
                arrowprops=dict(arrowstyle="->", linewidth=1))

    # Plot spline activations
    line1, = ax.plot(x, y_soft, linewidth=3, label="Spline (K=5)")
    line3, = ax.plot(x, y_sat,  linewidth=3.0, linestyle="-.",  label="Spline (K=7)")
    line2, = ax.plot(x, y_bump, linewidth=3.0, linestyle="--", label="Spline (K=9)")


    # Add knots to EACH line (no legend entries)
    ax.scatter(xk_soft, yk_soft, s=26, marker="o", color=line1.get_color(), zorder=3, label="_nolegend_")
    ax.scatter(xk_bumpy, yk_bumpy, s=26, marker="o", color=line2.get_color(), zorder=3, label="_nolegend_")
    ax.scatter(xk_sat, yk_sat,  s=26, marker="o", color=line3.get_color(), zorder=3, label="_nolegend_")

    # Remove default spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Ticks
    ax.set_yticks([])
    ax.set_xticks([])
    ax.tick_params(direction="out", length=5)
    legend = ax.legend(
        frameon=False,
        fontsize=14,  # larger text
        handlelength=2.0,  # longer line icon
        labelspacing=0.6,
        loc="upper left"
    )
    ax.tick_params(direction="out", length=5)

    # Styling
    ax.set_title("Spline-style Activation", fontsize=15)
    ax.set_xlabel("Input (x)", fontsize=12)
    ax.set_ylabel("Output", fontsize=12)

    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig("spline.png", dpi=300)
    plt.show()
