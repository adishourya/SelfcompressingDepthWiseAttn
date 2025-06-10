from turtle import color
import manim
import os
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.cm as cm

font_style = dict(font_size=22, font="Simple Nerd Font")

def qbits(x, b=6.0, e=-8.0):
    b = b if b > 0 else 0
    x_scaled = x / np.exp2(e)
    x_clipped = np.clip(x_scaled, -1 * np.exp2(b - 1), np.exp2(b - 1) - 1)
    x_round = np.round(x_clipped)
    result = np.exp2(e) * x_round
    return (result, x_scaled, x_clipped, x_round), (-1 * np.exp2(b - 1), np.exp2(b - 1) - 1)



def kernel_to_image(kernel: np.ndarray, colormap: str = "gray", use_abs=True, gamma=1.0) -> np.ndarray:
    """
    Converts a 2D kernel or feature map to an RGB uint8 image using a specified colormap.
    
    Parameters:
        kernel (np.ndarray): 2D array of floats.
        colormap (str): Matplotlib colormap name (default: 'gray').
        use_abs (bool): Whether to apply absolute value to kernel before visualization.
        gamma (float): Gamma correction factor (<1 brightens, >1 darkens).
    
    Returns:
        np.ndarray: RGB uint8 image.
    """
    if use_abs:
        kernel = np.abs(kernel)

    norm = (kernel - kernel.min()) / (kernel.max() - kernel.min() + 1e-8)
    
    # Contrast stretching
    vmin, vmax = np.percentile(norm, [1, 99])
    norm = np.clip((norm - vmin) / (vmax - vmin + 1e-8), 0, 1)

    # Gamma correction
    norm = norm ** gamma

    cmap = cm.get_cmap(colormap)
    colored = cmap(norm)[:, :, :3]
    return (colored * 255).astype(np.uint8)


class compressKernels(manim.Scene):
    def construct(self):
        title = manim.Text("(Ir)reversible Forgetting in Compressing Convolution kernels", **font_style)
        title.to_edge()
        self.play(manim.Write(title), runtime=0.3)
        self.play(manim.ScaleInPlace(title, 0.3), runtime=0.1)
        self.play(title.animate.to_edge(manim.DR), runtime=0.1)
        
        img_path = "./wall.png"
        img_path2 = "./feesh.png"
        img = manim.ImageMobject(img_path).scale(0.2)
        img2 = manim.ImageMobject(img_path2).scale(0.1)

        np_img = plt.imread(img_path)
        np_img = 0.2 * np_img[:, :, 0] + 0.7 * np_img[:, :, 1] + 0.1 * np_img[:, :, 2]


        np_img2 = plt.imread(img_path2)
        np_img2 = 0.2 * np_img2[:, :, 0] + 0.7 * np_img2[:, :, 1] + 0.1 * np_img2[:, :, 2]

        img_label = manim.Text("Input Image", **font_style)
        img_label2 = manim.Text("Pretend its a brick wall :)\nand the batch happens to have similar examples", **font_style)
        img_label.next_to(img, manim.DOWN)
        img_label2.next_to(img_label, manim.DOWN)
        img_g = manim.Group(img, img_label)
        
        # add image to screen
        self.play(manim.FadeIn(img_g))
        self.play(manim.FadeIn(img_label2))
        self.wait(1)
        self.play(manim.FadeOut(img_label2))
        self.wait(1)
        
        # move the image to write
        self.play(manim.ScaleInPlace(img_g, 0.5))
        self.play(img_g.animate.to_edge(manim.LEFT))
        self.wait(1)


        vertical_kernel = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=np.float32)

        horizontal_kernel = np.array([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=np.float32)

        diagonal_45_kernel = np.array([
            [ 0,  1,  2],
            [-1,  0,  1],
            [-2, -1,  0]
        ], dtype=np.float32)

        conv1 = signal.convolve2d(np_img, vertical_kernel)
        conv2 = signal.convolve2d(np_img, horizontal_kernel)
        conv3 = signal.convolve2d(np_img, diagonal_45_kernel)


        conv1_2 = signal.convolve2d(np_img2, vertical_kernel)
        conv2_2 = signal.convolve2d(np_img2, horizontal_kernel)
        conv3_2 = signal.convolve2d(np_img2, diagonal_45_kernel)

        # Kernels (grayscale)
        kernel1_img = kernel_to_image(vertical_kernel)
        kernel2_img = kernel_to_image(horizontal_kernel)
        kernel3_img = kernel_to_image(diagonal_45_kernel)

        # Outputs (colorized)
        from_kernel1 = kernel_to_image(conv1, colormap='Oranges')
        from_kernel2 = kernel_to_image(conv2, colormap='Greens')
        from_kernel3 = kernel_to_image(conv3, colormap='Blues')


        from_kernel1_2 = kernel_to_image(conv1_2, colormap='Oranges')
        from_kernel2_2 = kernel_to_image(conv2_2, colormap='Greens')
        from_kernel3_2 = kernel_to_image(conv3_2, colormap='Blues')

        kernel1 = manim.ImageMobject(kernel1_img)
        kernel2 = manim.ImageMobject(kernel2_img)
        kernel3 = manim.ImageMobject(kernel3_img)

        from_kernel1 = manim.ImageMobject(from_kernel1)
        from_kernel2 = manim.ImageMobject(from_kernel2)
        from_kernel3 = manim.ImageMobject(from_kernel3)

        from_kernel1_2 = manim.ImageMobject(from_kernel1_2)
        from_kernel2_2 = manim.ImageMobject(from_kernel2_2)
        from_kernel3_2 = manim.ImageMobject(from_kernel3_2)

        # Resize images
        kernel1.height = kernel2.height = kernel3.height = 2
        from_kernel1.height = from_kernel2.height = from_kernel3.height = 2
        from_kernel1_2.height = from_kernel2_2.height = from_kernel3_2.height = 2

        kernel1_copy = kernel1.copy()

        # Labels for kernels
        label1 = manim.Text("Vertical Filter", **font_style, color=manim.PINK).scale(0.5)
        label2 = manim.Text("Horizontal Filter", **font_style, color=manim.GREEN).scale(0.5)
        label3 = manim.Text("Diagonal Filter", **font_style, color=manim.BLUE).scale(0.5)

        # Labels for outputs
        label1_out = manim.Text("Vertical Out", **font_style, color=manim.PINK).scale(0.5)
        label2_out = manim.Text("Horizontal Out", **font_style, color=manim.GREEN).scale(0.5)
        label3_out = manim.Text("Diagonal Out", **font_style, color=manim.BLUE).scale(0.5)

        # Group each image with its label
        kernel1_group = manim.Group(kernel1, label1).arrange(manim.UP, buff=0.2)
        kernel2_group = manim.Group(kernel2, label2).arrange(manim.UP, buff=0.2)
        kernel3_group = manim.Group(kernel3, label3).arrange(manim.UP, buff=0.2)

        from_kernel1_group = manim.Group(from_kernel1, label1_out).arrange(manim.DOWN, buff=0.2)
        from_kernel2_group = manim.Group(from_kernel2, label2_out).arrange(manim.DOWN, buff=0.2)
        from_kernel3_group = manim.Group(from_kernel3, label3_out).arrange(manim.DOWN, buff=0.2)

        # Arrange in rows
        kernels_g = manim.Group(kernel1_group, kernel2_group, kernel3_group).arrange(manim.RIGHT, buff=1.0)
        from_kernels_g = manim.Group(from_kernel1_group, from_kernel2_group, from_kernel3_group).arrange(manim.RIGHT, buff=1.0)

        # Animate
        commentary1 = manim.Text("Sample Filter Banks",**font_style).to_edge(manim.UP).scale(0.6)
        self.add(commentary1)
        self.play(manim.FadeIn(kernels_g))
        self.play(kernels_g.animate.shift(manim.UP * 1.5))
        self.wait(1)

        horizontal_title = manim.Text("b=3.5,e=-3.5",**font_style,color=manim.GREEN)
        horizontal_title.next_to(kernel2,manim.DOWN)
        horizontal_tile_update = manim.Text("b=5.2,e=-6.1",**font_style,color=manim.GREEN)
        horizontal_tile_update.move_to(horizontal_title.get_center())

        diagonal_title = manim.Text("b=3.5,e=-3.5",**font_style,color=manim.BLUE)
        diagonal_title.next_to(kernel3,manim.DOWN)
        diagonal_title_update = manim.Text("b=4.1,e=-4.2",**font_style,color=manim.BLUE)
        diagonal_title_update.move_to(diagonal_title.get_center())


        faint_title = manim.Text("b=3.5,e=-3.5",**font_style,color=manim.ORANGE)
        faint_title_update = manim.Text("b=0,e=-2.5",**font_style,color=manim.ORANGE)
        faint_title_update2 = manim.Text("b=-3.5,e=-3.5",**font_style,color=manim.ORANGE)
        faint_title.next_to(kernel1,manim.DOWN)
        faint_title_update.move_to(faint_title.get_center())
        faint_title_update2.move_to(faint_title.get_center())
        
        # tie quant func to each conv kernel
        commentary2 = manim.Text("Broadcast Quantization Function to each kernel",**font_style).to_edge(manim.UP).scale(0.6)
        self.play(manim.Transform(commentary1,commentary2))

        self.play(manim.FadeIn(faint_title),run_time=0.5)
        self.play(manim.FadeIn(horizontal_title),run_time=0.5)
        self.play(manim.FadeIn(diagonal_title),run_time=0.5)
        self.wait(1)

        # output of convolution kernels
        commentary3 = manim.Text("Output from Quantized Conv Kernel",**font_style).to_edge(manim.UP).scale(0.6)
        self.play(manim.Transform(commentary1,commentary3))
        self.play(manim.FadeIn(from_kernels_g))
        self.play(from_kernels_g.animate.shift(manim.DOWN * 2))
        self.wait(2)
        
        # improve quant for relevant filters
        commentary4 = manim.Text("During training, additional bit depth may be allocated to kernels that generate stronger responses",**font_style).to_edge(manim.UP).scale(0.6)
        self.play(manim.Transform(commentary1,commentary4))
        self.play(manim.Indicate(horizontal_title,color=manim.GREEN), manim.Indicate(diagonal_title,color=manim.BLUE))
        self.wait(0.5)
        self.play(manim.Transform(horizontal_title,horizontal_tile_update),manim.Transform(diagonal_title,diagonal_title_update))
        self.wait(2)

        # forget the kernel
        commentary5 = manim.Text("And might reduce Bit Depth for other Conv kernels",**font_style).to_edge(manim.UP).scale(0.6)
        self.play(manim.Transform(commentary1,commentary5))
        self.play(manim.Indicate(faint_title))
        self.play(manim.Transform(faint_title,faint_title_update))
        self.wait(1)

        commentary6 = manim.Text("If the bit depth is reduced to zero, the entire kernel will be pruned.'Forgetting'",**font_style).to_edge(manim.UP).scale(0.6)
        self.play(manim.Transform(commentary1,commentary6))
        self.play(manim.Indicate(kernel1,color=manim.GRAY_BROWN), manim.Indicate(from_kernel1,color=manim.GRAY_BROWN))
        self.play(manim.FadeOut(kernel1), manim.FadeOut(from_kernel1))
        self.wait(1)


        commentary7 = manim.Text("More Generic Examples in the next batch",**font_style).to_edge(manim.UP).scale(0.6)
        self.play(manim.Transform(commentary1,commentary7))
        img2.move_to(img.get_center())
        self.play(manim.Indicate(img))
        self.play(manim.Transform(img,img2))
        new_batch_label = manim.Text("New Batch",**font_style).scale(0.6).move_to(img_label)
        self.play(manim.Transform(img_label,new_batch_label))

        from_kernel1_2.move_to(from_kernel1.get_center())
        from_kernel2_2.move_to(from_kernel2.get_center())
        from_kernel3_2.move_to(from_kernel3.get_center())

        commentary8 = manim.Text("For Few Iterations the loss is expected to be noisy",**font_style).to_edge(manim.UP).scale(0.6)
        self.play(manim.Transform(commentary1,commentary8))
        self.play(manim.Transform(from_kernel2, from_kernel2_2),
                  manim.Transform(from_kernel3,from_kernel3_2)) 
        
        self.wait(1)
        kernel1_copy.move_to(kernel1.get_center())

        commentary9 = manim.Text("However the bit depth can be recoverd 'Reversible Forgetting'",**font_style).to_edge(manim.UP).scale(0.6)
        self.play(manim.Transform(commentary1,commentary9))
        self.play(manim.Transform(faint_title,faint_title_update2),
                  manim.FadeIn(from_kernel1, from_kernel1_2),
                  manim.FadeIn(kernel1, kernel1_copy),
                  run_time=3)


        self.wait(5)


if __name__ == "__main__":
    os.system("manim -qh --resolution 1920,1080 tying_conv.py compressKernels")
