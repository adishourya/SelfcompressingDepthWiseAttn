import cutlass.cute as cute

def unravel(idx, shape):
    """Unravel a flat index into hierarchical coordinate matching shape."""
    if isinstance(shape, int):
        return idx
    elif isinstance(shape, tuple):
        # Assume 2D nested tuple
        s0, s1 = shape
        return (idx % s0, idx // s0)
    else:
        raise ValueError("Only handles int or tuple of 2 ints")

def flatten_shape(shape):
    """Return total number of elements in a (possibly nested) shape."""
    if isinstance(shape, int):
        return shape
    elif isinstance(shape, tuple):
        total = 1
        for s in shape:
            total *= flatten_shape(s)
        return total
    else:
        raise ValueError("Unsupported shape")

def visualize_layout(L: cute.Layout):
    shape = L.shape
    stride = L.stride

    print("="*15)
    print(f"Layout: shape={shape}, stride={stride}")

    if len(shape) == 1:
        # Rank-1 layout (vector, possibly nested mode)
        length = flatten_shape(shape[0])
        row = []
        for i in range(length):
            coord = (unravel(i, shape[0]),)
            idx = cute.crd2idx(coord, L)
            row.append(f"{int(idx):3}")
        print(" ".join(row))
    elif len(shape) == 2:
        # Rank-2 layout (matrix, possibly nested modes)
        rows = flatten_shape(shape[0])
        cols = flatten_shape(shape[1])
        for i in range(rows):
            row_vals = []
            for j in range(cols):
                i_coord = unravel(i, shape[0])
                j_coord = unravel(j, shape[1])
                idx = cute.crd2idx((i_coord, j_coord), L)
                row_vals.append(f"{int(idx):3}")
            print(" ".join(row_vals))
    else:
        raise NotImplementedError("Only supports rank-1 or rank-2 layouts")

    print("="*15)

# Example usage
if __name__ == "__main__":
    print("Matrix Layout:")
    L = cute.make_layout(shape=(2, (4, 4)), stride=(32, (4, 1)))
    visualize_layout(L)

    print("\nFlipped Matrix Layout:")
    L2 = cute.make_layout(shape=((4, 4), 2), stride=(16, (4, 1)))
    visualize_layout(L2)

    print("\nRank-1 Vector Layout:")
    L3 = cute.make_layout(shape=((4, 2),), stride=((2, 1),))
    visualize_layout(L3)
