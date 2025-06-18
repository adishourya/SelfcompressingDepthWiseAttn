# https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/01_layout.md
# Layout is a bijective function that maps from cordinate space to an index space (memory space) which is 1d


# def visualize_layout(L:cute.Layout):
#     shape = L.shape
#     # the modes of the shape would always be 2
#     # we will only print for 2d layouts
#     assert len(shape) == 2, "only prints 2d layouts"
#     # if mode1 is nested say shape is ((a,b), c)
#     # then we need to generate hierarchical indexes for mode1
#     # suppose index is 3,4.. then we would have to generate hierarchical index for 3
#     # which can be done as ((3%shape[0], 3//shape[0]) , 4)
#     # and now we can calculate the idx with cute.crd2idx(some_idx,tile_mode1)
#



import cutlass
import cutlass.cute as cute
from cutlass.cute.core import shape
import cutlass.torch as cltorch
from vis_layout import visualize_layout

import torch

def intro():
    # by default it is row major or Right Layout
    # that is (m,n):(n,1)
    a = torch.arange(16).reshape(4,4)
    print(a,a.stride())

    # we can change it to someother layout
    # here (m,n): (1,m) i.e column major or left layout
    # we can change the memory layout.. all we need is the size and stride
    # to reach the canonical form
    # layout then is given by multi_dim_idx dot stride
    b = torch.arange(16).reshape(4,4).as_strided(size=(4,4),
                                                 stride= (1,4))
    print(b,b.stride())


# constructing a layout

@cute.jit
def example_constructing_layouts():
    # this is allowed ... interleaving
    d2xd4_a = cute.make_layout(shape = (8,8),
                               stride = (12,1)
                               ) 
    visualize_layout(d2xd4_a)
    idx = cute.crd2idx((4,3),d2xd4_a)
    # expectation (4,3) o (12,1) = 48 + 3 = 51
    cute.printf(d2xd4_a,idx)
    #----------------------------------------------------
    # we can also make arbitarily nested layout [Tiling]
    # this can be read as for mode column we dont tile.. we have a stride of 3
    # (8,2) suggests a tile of size 8 and we have 2 of those
    # associated stride suggests we will move by 2 within the tile
    # and the offset to reach the other tile is 1000
    tile_mode1 = cute.make_layout(shape= (4,(8,2)), # 2,16
                                stride=(3,(2,1000))
                                )
    visualize_layout(tile_mode1)
    idx = cute.crd2idx((1,13),tile_mode1)
    # expectation : (1,13) o (3,(2,1000)) ?
    # 13 -> position in tile, wich tile
    # 13 -> (13%8 , 13//8) -> (5,1)
    # (1,(5,1)) o (3,(2,1000)) -> 3 + 10 + 1000 = 1013
    cute.printf(tile_mode1,idx)
    #----------------------------------------------------

    # now this can be read as we have a tile in mode 1
    # 2 tiles each of size 8.
    # we move within the tile by a stride of 8
    # and to reach the next tile we go 1000
    # and to move to the other mode we have a stride of 3
    tile_mode2 = cute.make_layout(shape= ((8,2),4), # 16,2
                                stride=((2,1000),3)
                                )
    visualize_layout(tile_mode2)
    idx = cute.crd2idx((1,13),tile_mode2)
    # suppose we want 5th row and 2nd column : (6,1)
    # then 6 is at 6th position of the first tile -> (6,0)
    # so our hierarchical index is ((6,0),1)
    # ((6,0),1) o ((2,1000),3) = 6*2 + 1*3 = 15
    #-------------------
    tile2d = cute.make_layout(shape= ((4,2), (4,2)),
                              stride= ((4,32),(1,16))
                              )
    cute.printf(tile2d)
    visualize_layout(tile2d)
    #--------


@cute.jit
def example_coalescing_layouts():
    layout_rm = cute.make_layout(shape= (4,(8,2)),
                               stride=(1,(4,32))
                               )
    # flattened 3d layout
    co_layout_rm = cute.coalesce(layout_rm)

    visualize_layout(layout_rm)
    cute.printf(layout_rm,co_layout_rm)

    #-----------------------------------------
    layout_cm = cute.make_layout(shape = (4,(8,2)),
                               stride= (16, (1,8))
                               )
    visualize_layout(layout_cm)
    co_layout_cm = cute.coalesce(layout_cm)
    cute.printf(layout_cm, co_layout_cm)
    #------------------------------------------
    # column major 2d tile
    tile2d_cm = cute.make_layout(shape= ((4,2), (4,2)),
                              stride= ((1,4),(8,32))
                              )
    visualize_layout(tile2d_cm)
    co_tile2d_cm = cute.coalesce(tile2d_cm)
    cute.printf(tile2d_cm,co_tile2d_cm)
    #------------------------------------------
    # row major 2d tile
    tile2d_rm = cute.make_layout(shape= ((4,2), (4,2)),
                                 stride = ((8,32), (1,4))
                                 )
    visualize_layout(tile2d_rm)
    co_tile2d_rm = cute.coalesce(tile2d_rm)
    cute.printf(tile2d_rm,co_tile2d_rm)
    #------------------------------------------
    another_tile2d_rm = cute.make_layout(shape = ((2,2), (4,4)), 
                                         stride= ((32,4),(4,1))
                                         )
    visualize_layout(another_tile2d_rm)
    pass


if __name__ == "__main__":
    # intro()
    # example_constructing_layouts()
    example_coalescing_layouts()

