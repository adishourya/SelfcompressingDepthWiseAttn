import cutlass.cute as cute

def unravel(idx, shape):
    """Unravel a flat index into hierarchical coordinate matching shape."""
    if isinstance(shape, int):
        return idx
    elif isinstance(shape, tuple) and len(shape) == 2:
        s0, s1 = shape
        return (idx % s0, idx // s0)
    else:
        raise ValueError("Only handles up to 2-level nested shapes")

def visualize_layout(L: cute.Layout):
    shape = L.shape
    assert len(shape) <= 2, "Only supports 2D layouts"

    rows, cols = shape
    rows_flat = (rows if isinstance(rows, int) else rows[0] * rows[1])
    cols_flat = (cols if isinstance(cols, int) else cols[0] * cols[1])

    print(f"Layout: shape={shape}, stride={L.stride}")
    for i in range(rows_flat):
        row = []
        for j in range(cols_flat):
            i_hier = unravel(i, rows)
            j_hier = unravel(j, cols)
            idx = cute.crd2idx((i_hier, j_hier), L)
            row.append(f"{int(idx):3}")
        print(" ".join(row))

# Example usage
if __name__ == "__main__":
    L = cute.make_layout(shape=(2, (4, 4)), stride=(32, (4, 1)))
    visualize_layout(L)

    print("\n")

    L2 = cute.make_layout(shape=((4, 4), 2), stride=(16, (4, 1)))
    visualize_layout(L2)
