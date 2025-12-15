import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Polynomial-style activations (max degree 5)
# -------------------------
def poly_deg3(x):
    # Cubic: strong curvature, ReLU-like but aggressive
    # f(x) = 1.10x - 0.045x^3
    return 1.10 * x - 0.02 * x**3


def poly_deg4(x):
    # Quartic: S-skewed, asymmetric activation
    # f(x) = 0.75x + 0.06x^2 - 0.015x^3 - 0.0020x^4
    return 0.75 * x + 0.06 * x**2 - 0.015 * x**3 - 0.0020 * x**4


def poly_deg5(x):
    # Quintic: smooth gated activation
    # f(x) = 0.95x - 0.035x^3 + 0.0018x^5
    return 0.95 * x - 0.035 * x**3 + 0.0018 * x**5





if __name__ == "__main__":
    # Data
    x = np.linspace(-5, 5, 800)

    y3 = poly_deg3(x)
    y4 = poly_deg4(x)
    y5 = poly_deg5(x)

    # Figure (your style)
    fig, ax = plt.subplots(figsize=(6, 5))

    limit = 5
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    ax.annotate(
        "", xy=(limit, 0), xytext=(-limit, 0),
        arrowprops=dict(arrowstyle="->", linewidth=1)
    )
    ax.annotate(
        "", xy=(0, limit), xytext=(0, -limit),
        arrowprops=dict(arrowstyle="->", linewidth=1)
    )

    # Plot polynomials (legend shows numeric meaning)
    ax.plot(x, y3, linewidth=3,
            label=r"Poly (deg=3)")
    ax.plot(x, y4, linewidth=3.0, linestyle="--",
            label=r"Poly (deg=4)")
    ax.plot(x, y5, linewidth=3.0, linestyle="-.",
            label=r"Poly (deg=5)")

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
    ax.set_title("Polynomial-style Activation", fontsize=15)
    ax.set_xlabel("Input (x)", fontsize=12)
    ax.set_ylabel("Output", fontsize=12)
    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig("polynomial.png", dpi=300)
    plt.show()
