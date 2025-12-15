import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Fourier-style activation
# -------------------------
def fourier_activation(
    x,
    N,
    w,
    gain,
    bound,
    mode="odd",
    a=None,
    b=None,
):
    """
    Fourier-style activation:
      core(x) = gain*x + Σ a_n sin(n*w*x) [+ Σ b_n cos(n*w*x)]
    then softly bounded with tanh for stability.
    """
    x = np.asarray(x, dtype=float)

    # Default decaying coefficients (keeps it activation-like)
    if a is None:
        a = np.array([0.9 / (n**1.15) for n in range(1, N + 1)])
    if b is None:
        b = np.array([0.5 / (n**1.15) for n in range(1, N + 1)])

    core = gain * x

    # Sine harmonics (odd component)
    for n in range(1, N + 1):
        core += a[n - 1] * np.sin(n * w * x)

    # Cosine harmonics (even / asymmetry)
    if mode == "asym":
        for n in range(1, N + 1):
            core += b[n - 1] * np.cos(n * w * x)

    # Gated variant: suppress oscillations for x < 0
    if mode == "gated":
        gate = 0.30 + 0.70 * np.tanh(0.9 * x)
        core = gain * x + gate * (core - gain * x)

    # Soft bounding (keeps it an activation)
    return bound * np.tanh(core / bound)


if __name__ == "__main__":
    x = np.linspace(-5, 5, 1400)

    # -------------------------
    # Three DISTINCT Fourier activations
    # -------------------------

    # (1) N = 1 : almost monotone, very activation-like
    y1 = fourier_activation(
        x,
        N=1,
        w=0.8,
        gain=0.95,
        bound=3.0,
        mode="odd"
    )

    # (2) N = 3 : asymmetric, richer curvature
    y3 = fourier_activation(
        x,
        N=3,
        w=1.2,
        gain=0.60,
        bound=2.6,
        mode="asym"
    )

    # (3) N = 5 : expressive but gated (less periodic-looking)
    a5 = np.array([1.4, 1.0, 0.7, 0.45, 0.30])  # slower decay => more detail

    y5 = fourier_activation(
        x,
        N=5,
        w=0.95,  # higher frequency than N=3
        gain=0.45,  # less linear dominance
        bound=3.8,  # larger dynamic range
        mode="asym",
        a=a5
    )

    # -------------------------
    # Figure (your style)
    # -------------------------
    fig, ax = plt.subplots(figsize=(6, 5))

    limit = 5
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    ax.annotate("", xy=(limit, 0), xytext=(-limit, 0),
                arrowprops=dict(arrowstyle="->", linewidth=1))
    ax.annotate("", xy=(0, limit), xytext=(0, -limit),
                arrowprops=dict(arrowstyle="->", linewidth=1))

    ax.plot(x, y1, linewidth=3,
            label="Fourier (N=1)")
    ax.plot(x, y3, linewidth=3.0, linestyle="--",
            label="Fourier (N=3)")
    ax.plot(x, y5, linewidth=3.0, linestyle="-.",
            label="Fourier (N=5)")

    # Remove default spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Ticks
    # ax.set_xticks(np.arange(-5, 6, 2.5))
    # ax.set_yticks(np.arange(-5, 6, 2.5))
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
    ax.set_title("Fourier-style Activation Functions", fontsize=15)
    ax.set_xlabel("Input (x)", fontsize=12)
    ax.set_ylabel("Output", fontsize=12)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig("Fourier.png", dpi=300)
    plt.show()
