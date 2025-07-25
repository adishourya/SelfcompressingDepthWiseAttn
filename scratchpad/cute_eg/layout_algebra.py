# https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/01_layout.md
# Layout is a bijective function that maps from cordinate space to an index space (memory space) which is 1d

import cutlass
import cutlass.cute as cute
from vis_layout import visualize_layout
import torch


def intro():
    # by default it is row major or Right Layout
    # that is (m,n):(n,1)
    a = torch.arange(16).reshape(4, 4)
    print(a, a.stride())

    # we can change it to someother layout
    # here (m,n): (1,m) i.e column major or left layout
    # we can change the memory layout.. all we need is the size and stride
    # to reach the canonical form
    # layout then is given by multi_dim_idx dot stride
    b = torch.arange(16).reshape(4, 4).as_strided(size=(4, 4), stride=(1, 4))
    print(b, b.stride())


# constructing a layout


@cute.jit
def example_constructing_layouts():
    # this is allowed ... interleaving
    d2xd4_a = cute.make_layout(shape=(8, 8), stride=(12, 1))
    visualize_layout(d2xd4_a)
    idx = cute.crd2idx((4, 3), d2xd4_a)
    # expectation (4,3) o (12,1) = 48 + 3 = 51
    cute.printf(d2xd4_a, idx)
    # ----------------------------------------------------
    # we can also make arbitarily nested layout [Tiling]
    # this can be read as for mode column we dont tile.. we have a stride of 3
    # (8,2) suggests a tile of size 8 and we have 2 of those
    # associated stride suggests we will move by 2 within the tile
    # and the offset to reach the other tile is 1000
    tile_mode1 = cute.make_layout(
        shape=(4, (8, 2)),  # 2,16
        stride=(3, (2, 1000)),
    )
    visualize_layout(tile_mode1)
    idx = cute.crd2idx((1, 13), tile_mode1)
    # expectation : (1,13) o (3,(2,1000)) ?
    # 13 -> position in tile, wich tile
    # 13 -> (13%8 , 13//8) -> (5,1)
    # (1,(5,1)) o (3,(2,1000)) -> 3 + 10 + 1000 = 1013
    cute.printf(tile_mode1, idx)
    # ----------------------------------------------------

    # now this can be read as we have a tile in mode 1
    # 2 tiles each of size 8.
    # we move within the tile by a stride of 8
    # and to reach the next tile we go 1000
    # and to move to the other mode we have a stride of 3
    tile_mode2 = cute.make_layout(
        shape=((8, 2), 4),  # 16,2
        stride=((2, 1000), 3),
    )
    visualize_layout(tile_mode2)
    idx = cute.crd2idx((1, 13), tile_mode2)
    # suppose we want 5th row and 2nd column : (6,1)
    # then 6 is at 6th position of the first tile -> (6,0)
    # so our hierarchical index is ((6,0),1)
    # ((6,0),1) o ((2,1000),3) = 6*2 + 1*3 = 15
    # -------------------
    tile2d = cute.make_layout(shape=((4, 2), (4, 2)), stride=((4, 32), (1, 16)))
    cute.printf(tile2d)
    visualize_layout(tile2d)
    # --------


@cute.jit
def example_coalescing_layouts():
    # cpp example
    l1 = cute.make_layout(shape =(2,(1,6)),
                          stride= (1,(6,2)))
    l1_coalesced = cute.coalesce(l1)
    visualize_layout(l1)
    # visualize_layout(l1_coalesced)
    cute.printf(l1_coalesced)
    
    # 1d tile in row mode
    tile1d = cute.make_layout(
        shape=(4, (8, 2)),
        stride=(1, (4, 32)),  # column major
        # stride= (16, (1,8)) # row major
    )
    # flattened 3d layout
    co_tile1d = cute.coalesce(tile1d)

    # visualize_layout(tile1d)
    # cute.printf(tile1d, co_tile1d)

    # ------------------------------------------
    # tile of shape (4,4)
    tile2d = cute.make_layout(
        shape=((4, 2), (4, 2)),
        stride=((1, 4), (8, 32)),  # column major
        # stride= ((1,4),(9,32)) # off by 1 column major
        # stride = ((8,32), (1,4)) # row major
    )
    co_tile2d = cute.coalesce(tile2d)
    # visualize_layout(tile2d)
    # cute.printf(tile2d, co_tile2d)

    # ------------------------------------------
    # a tile of shape 2,4
    rect_tile2d = cute.make_layout(
        shape=((2, 2), (4, 4)),
        # stride= ((1,1000),(2,100))
        stride=((1, 2), (4, 16)),  # column major
        # stride = ((16,32),(1,4)) # row major
    )
    # visualize_layout(rect_tile2d)
    # co_rect_tile2d = cute.coalesce(rect_tile2d)
    # cute.printf(rect_tile2d, co_rect_tile2d)


@cute.jit
def example_vector_layout():
    vec_a = cute.make_layout(shape=(8), stride=(1))
    vec_b = cute.make_layout(shape=((4, 2),), stride=((1, 100),))
    cute.printf(vec_a)
    visualize_layout(vec_b)


@cute.jit
def example_matrix_layout():
    matrix_a = cute.make_layout(shape=((4, 3), 2), stride=((10, 100), 1))
    # cute.printf(matrix_a)
    visualize_layout(matrix_a)

# @cute.jit
def example_cordinate_mapping():
    # say we have a shape of (3,(2,3))
    d0, d1 , d2 = 3,2,3
    # then what would be the 1d cordinate of it.
    # this is just 1d cordinate not the offset.
    # stride is not used in calculating cordinate mapping.
    # here the range is 3,6 -> 18
    def mapping(one_d):
        """
            This is how you would go from 1d mapping to natural mapping
        """
        i = one_d % d0
        one_d = one_d // d0 # drop component
        j = one_d % d1
        one_d = one_d // d1
        k = one_d
        return f"({i},({j},{k}))"
        

    for i in range(18):
        print(f"one_d = {i} natural = {mapping(i)}")
        


@cute.jit
def example_composition_layouts():
    A = cute.make_layout(shape=(6,2),stride=(100,5))
    B = cute.make_layout(shape=(2,3),stride=(3,1))

    # A = cute.make_layout(shape=(20,1),stride=(10,1))
    # B = cute.make_layout(shape=(4,3),stride=(3,1))
    AoB = cute.composition(A,B)
    # visualize_layout(B)
    # visualize_layout(A)
    # visualize_layout(AoB)

    # so the size of the layout is the same 12 and 12.
    # R = A o B => R(one_d) = A(B(one_d))
    # B(one_d) also returns one_d
    # then finally A(one_d) would be R(one_d)

    a_d0, a_d1 = A.shape
    a_s0, a_s1 = A.stride
    b_d0,b_d1 = B.shape
    b_s0,b_s1 = B.stride

    @cute.jit
    def get_idx():
        for i in range(12):
            one_d =i
            b_i = one_d % b_d0
            one_d = one_d // b_d0
            b_j = one_d % b_d1

            # now we can fetch
            b_out = cute.crd2idx((b_i,b_j),B)
            b_our = b_i*b_s0 + b_j*b_s1

            # this is one_d for a
            one_d = b_out
            a_i = one_d % a_d0
            one_d = one_d//a_d0
            a_j = one_d % a_d1

            # now we can fetch from A
            a_our = a_i * a_s0 + a_j * a_s1
            a_out = cute.crd2idx((a_i,a_j),A)

            print(f"R({i}) =R({b_i},{b_j}) =  A(B({b_i},{b_j})) = A({b_out}|{b_our}) = A({a_i},{a_j}) = {a_out}|{a_our}")
            # cute.printf((b_i,b_j),b_out,(a_i,a_j),a_out)
            # cute.printf((a_our,a_out), (b_our,b_out))

    get_idx()

@cute.jit
def example_composition_layout2():
    """
        the resultant layout still remains to be mystery to me.
        what happens when B is row major order. how do i calculate the layout by hand.
    """
    A = cute.make_layout(shape=(6,2),stride=(100,5))
    B = cute.make_layout(shape=(2,3),stride=(3,1))
    AoB = cute.composition(A,B)
    cute.printf(AoB)


@cute.jit
def example_bymode_composition():
    """
    Demonstrates by-mode composition using a tiler
    """
    # Define the original layout A
    # A_dyn = cute.make_layout(
    #     (cutlass.Int32(12), (cutlass.Int32(4), cutlass.Int32(8))), 
    #     stride=(cutlass.Int32(59), (cutlass.Int32(13), cutlass.Int32(1)))
    # )

    A = cute.make_layout(
        shape =(12,(4,8)),
        stride = (59,(13,1))
    )
    visualize_layout(A)

    # Define the tiler for by-mode composition
    tiler = (3, 8) # Apply 3:1 to mode-0 and 8:1 to mode-1

    # we can also make a tiler... in python its just a tuple of layouts
    # tiler2 = (
    #     cute.make_layout(shape=(3,), stride=(2,)),
    #     cute.make_layout(shape=(8,),stride=(2,))
    #                        )

    # Apply by-mode composition
    # (12, (4,8)) : (59, (13,1)) o (3,8)
    # obviously the result will take the values from A but will be the shape of tiler
    result = cute.composition(A, tiler)
    visualize_layout(result)

    # Print static and dynamic information
    print(">>> Tiler:", tiler)
    print(">>> By-mode Composition Result:", result)
    
@cute.jit
def example_complement():
    # l1 = cute.make_layout(shape=(4,(2,3)),
    #                       stride=(2,(1,8)))

    l1 = cute.make_layout(shape=((2,4),3),
                          stride=((1,6),2))
    visualize_layout(l1)

@cute.jit
def example_logical_divide():
    """
    Demonstrates 1D logical divide
    """
    # Define the original layout
    layout = cute.make_layout((4, 2, 3), stride=(2, 1, 8))  # (4,2,3):(2,1,8)
    # visualize_layout(layout)
    
    # Define the tiler
    tiler = cute.make_layout(4, stride=2)  # Apply to layout 4:2
    
    # Apply logical divide
    result = cute.logical_divide(layout, tiler=tiler)
    # visualize_layout(tiler)
    
    # Print results
    print(">>> Layout:", layout)
    print(">>> Tiler :", tiler)
    print(">>> Logical Divide Result:", result)
    cute.printf(">?? Logical Divide Result: {}", result)


if __name__ == "__main__":
    # intro()
    # example_constructing_layouts()
    # example_vector_layout()
    # example_matrix_layout()
    # example_cordinate_mapping()
    # example_coalescing_layouts()
    # example_composition_layouts()
    # example_composition_layout2()
    # example_bymode_composition()
    # example_logical_divide()
    example_complement()

    pass
