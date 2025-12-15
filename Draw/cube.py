import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import os

# Ensure the output directory exists
OUTPUT_DIR = "cube_outputs_hwc_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def draw_cube(ax, origin, size, color, edge_color):
    """
    Draws a single cube using Poly3DCollection.
    """
    x, y, z = origin
    s = size

    # Define the 8 vertices of the cube
    vertices = np.array([
        [x, y, z], [x + s, y, z], [x + s, y + s, z], [x, y + s, z],
        [x, y, z + s], [x + s, y, z + s], [x + s, y + s, z + s], [x, y + s, z + s]
    ])

    # Define the 6 faces (polygons) using the vertices
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # Bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # Top
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back
        [vertices[1], vertices[2], vertices[6], vertices[5]],  # Right
        [vertices[4], vertices[7], vertices[3], vertices[0]]  # Left
    ]

    # Create collection, setting edge color to black for distinct gaps
    cube = Poly3DCollection(faces, linewidths=1.0, edgecolors=edge_color, facecolors=color)
    ax.add_collection3d(cube)


# ==============================================================================
# CORE FUNCTION FOR DRAWING AXES BESIDES THE CUBE
# ==============================================================================
def add_custom_axes(ax, origin, length):
    """
    Adds custom H, W, C arrows (lines) and text labels to the plot.
    They are placed at 'origin' with a specific 'length'.
    Style: Black lines and black bold text.
    """
    ox, oy, oz = origin

    # Style settings for lines (quiver arrows) and text
    arrow_kw = dict(arrow_length_ratio=0.1, lw=2, color='black')
    text_kw = dict(fontsize=20, fontfamily='sans-serif', ha='center', va='center', color='black')
    label_offset = length * 1.05

    # --- Mapping dimensions to visual axes based on isometric view ---
    gap = 0.2  # Small gap to separate axes visually

    # C axis: Mapped to X-axis. Visually appears as the "left-down" line.
    ax.quiver(ox, oy+gap, oz, -length*0.95, 0, 0, **arrow_kw)
    ax.quiver(ox-length*0.95, oy + gap, oz, length*0.95, 0, 0, **arrow_kw)
    ax.text(ox - label_offset /2.3, oy + gap * 2.7, oz, "C", **text_kw)

    # W axis: Mapped to Y-axis. Visually appears as the "right-down" line (depth).
    ax.quiver(ox, 0, oz-gap, 0, +length*0.97, 0, **arrow_kw)
    ax.quiver(ox, 0+length*0.97, oz - gap, 0, -length*0.97, 0, **arrow_kw)
    ax.text(ox, 0 + label_offset/2.4, oz-gap*2.9, "W", **text_kw)


    # H axis: Mapped to Z-axis. Visually appears as the "far left" vertical line.
    ax.quiver(ox+gap, 0, oz, 0, 0, length, **arrow_kw)
    ax.quiver(ox + gap, 0, oz+length, 0, 0, -length, **arrow_kw)

    ax.text(ox + 2.7*gap, 0, oz + label_offset /2.1, "H", **text_kw)

    # ax.text(ox, oy, 1  , "H", **dict(text_kw, va='top'))
    print(" ox, oy, oz + label_offset =", ox, oy, oz + label_offset, oz)

# ==============================================================================


def save_cube_graph(orange_positions_list, filename):
    fig = plt.figure(figsize=(10, 10), facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='white')

    # Configuration
    grid_dim = 5
    step = 1.0
    cube_size = 0.93  # Slightly smaller than 1.0 to create gaps
    blue_color = '#0f4c81'
    orange_color = '#e07a1f'
    edge_color = 'black'  # Black edges for high contrast gaps

    # Draw the grid of cubes
    orange_pos_set = set(orange_positions_list)
    for x in range(grid_dim):
        for y in range(grid_dim):
            for z in range(grid_dim):
                origin = (x * step, y * step, z * step)
                if (x, y, z) in orange_pos_set:
                    color = orange_color
                else:
                    color = blue_color
                draw_cube(ax, origin, cube_size, color, edge_color)

    # Calculate grid limits
    grid_limit = grid_dim * step

    # ==========================================================================
    # SETUP AND CALL CUSTOM AXES
    # ==========================================================================
    # Define origin outside the cube grid (negative coordinates) so it sits "besides" it
    axis_origin = (5, 5, 0)
    axis_length = 6

    # Call the function defined above
    # add_custom_axes(ax, axis_origin, axis_length)

    axis_origin = (0, 0, 0)
    # Adjust plot limits to ensure the new axes and labels are visible.
    # We expand the lower bounds to accommodate the negative axis origin.
    buffer = 1.0
    ax.set_xlim([axis_origin[0] - buffer, grid_limit])
    ax.set_ylim([axis_origin[1] - buffer, grid_limit])
    ax.set_zlim([axis_origin[2] - buffer, grid_limit])

    # IMPORTANT: Calculate span and set box_aspect to ensure cubic proportions
    span_x = grid_limit - (axis_origin[0] - buffer)
    span_y = grid_limit - (axis_origin[1] - buffer)
    span_z = grid_limit - (axis_origin[2] - buffer)
    ax.set_box_aspect((span_x, span_y, span_z))
    # ==========================================================================
    axis_origin = (5, 5, 0)
    axis_length = 5

    # Call the function defined above
    add_custom_axes(ax, axis_origin, axis_length)

    # Standard isometric view angle
    ax.view_init(elev=30, azim=45)

    # Hide default Matplotlib axes
    ax.set_axis_off()
    ax.grid(False)
    plt.tight_layout(pad=5)

    # Save figure
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f"Saved image: {filepath}")


if __name__ == "__main__":
    # Define configurations
    configurations = [
        ([(4, 4, 3)], "config1_single.jpg"),
        ([(4, 4, 3), (3, 4, 3), (2, 4, 3), (1, 4, 3), (0, 4, 3)], "config2_row.jpg"),
        ([(4, i, j) for i in range(5) for j in range(5)], "config3_face.jpg"),
        ([(i, z, j) for i in range(5) for j in range(5) for z in range(5)], "config4_all.jpg")
    ]

    print("Starting image generation with HWC axes...")
    for pos_list, filename in configurations:
        save_cube_graph(pos_list, filename)
    print("Finished generation.")