import numpy as np
import matplotlib.pyplot as plt

# Activation functions
def relu(x):
    return np.maximum(0, x)

def swish(x, beta=1.0):
    return x / (1 + np.exp(-beta * x))


if __name__ == "__main__":
    # Data

    x = np.linspace(-5, 5, 600)

    # Compute activations
    y_relu = relu(x)
    y_swish_05 = swish(x, beta=0.5)
    y_swish_1 = swish(x, beta=1.0)
    y_swish_2 = swish(x, beta=2.0)

    # Figure
    fig, ax = plt.subplots(figsize=(6, 5))

    # Symmetric limits (origin centered)
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

    # Plot functions
    ax.plot(x, y_relu, linewidth=2.8, label="ReLU")
    # ax.plot(x, y_swish_05, linewidth=2, linestyle="--", label="Swish (β = 0.5)")
    # ax.plot(x, y_swish_1, linewidth=2, label="Swish (β = 1)")
    # ax.plot(x, y_swish_2, linewidth=2, linestyle=":", label="Swish (β = 2)")

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
        labelspacing=0.8,
        loc="upper left"
    )

    # Arrowed axes


    # Styling
    ax.set_title("ReLU Activation Functions", fontsize=15)
    ax.set_xlabel("Input (x)", fontsize=12)
    ax.set_ylabel("Output", fontsize=12)

    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig("relu.png", dpi=300)
    plt.show()

