
from scipy.ndimage import gaussian_filter
from turtle import color
import manim
import os
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
# import matplotlib.cm as cm  # not needed directly anymore
import torch
import torch.nn as nn

font_style = dict(font_size=22, font="Simple Nerd Font")
font_style2 = dict(font_size=12, font="Simple Nerd Font")

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
    """
    if use_abs:
        kernel = np.abs(kernel)

    # Normalize
    kmin, kmax = kernel.min(), kernel.max()
    norm = (kernel - kmin) / (kmax - kmin + 1e-8)

    # Contrast stretching
    vmin, vmax = np.percentile(norm, [1, 99])
    norm = np.clip((norm - vmin) / (vmax - vmin + 1e-8), 0, 1)

    # Gamma correction
    norm = norm ** gamma

    # Use modern colormap API
    cmap = plt.get_cmap(colormap)
    colored = cmap(norm)[..., :3]
    return (colored * 255).astype(np.uint8)


class ShufflingProblem(manim.Scene):
    def construct(self):
        title = manim.Text("Loss of Granularity in Quantized Pixel Shuffling", **font_style)
        title.to_edge()
        self.play(manim.Write(title), runtime=0.3)
        self.play(manim.ScaleInPlace(title, 0.3), runtime=0.1)
        self.play(title.animate.to_edge(manim.DR), runtime=0.1)

        # --- ONE input image only ---
        img_path = "./dolphin.jpg"
        img = manim.ImageMobject(img_path).scale(0.2)

        # grayscale (weighted)
        np_img = plt.imread(img_path)
        if np_img.ndim == 3:
            # handle RGBA as well
            if np_img.shape[2] == 4:
                np_img = np_img[:, :, :3]
            np_img = 0.2 * np_img[:, :, 0] + 0.7 * np_img[:, :, 1] + 0.1 * np_img[:, :, 2]
        np_img = np.asarray(np_img, dtype=np.float32)

        img_label = manim.Text("Input Image", **font_style)
        img_label.next_to(img, manim.DOWN)
        img_g = manim.Group(img, img_label)

        # add image to screen (left side)
        self.play(manim.FadeIn(img_g))
        self.play(manim.ScaleInPlace(img_g, 0.5))
        self.play(img_g.animate.to_edge(manim.LEFT), runtime=0.4)
        self.wait(0.3)

        # --- 4 kernels (now includes a zero kernel) ---
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

        rdiagonal_45_kernel = np.array([
            [ 2,  1,  2],
            [-1,  0,  1],
            [0, -1,  -2]
        ], dtype=np.float32)


        # Convolve with 'same' size so all feature maps share shape
        conv1 = signal.convolve2d(np_img, vertical_kernel, mode='same', boundary='fill')
        conv2 = signal.convolve2d(np_img, horizontal_kernel, mode='same', boundary='fill')
        conv3 = signal.convolve2d(np_img, diagonal_45_kernel, mode='same', boundary='fill')
        conv4 = signal.convolve2d(np_img, rdiagonal_45_kernel, mode='same', boundary='fill')

        # --- Visuals for kernels ---
        kernel1_img = kernel_to_image(vertical_kernel, colormap="gray")
        kernel2_img = kernel_to_image(horizontal_kernel, colormap="gray")
        kernel3_img = kernel_to_image(diagonal_45_kernel, colormap="gray")
        kernel4_img = kernel_to_image(rdiagonal_45_kernel, colormap="gray", use_abs=True)

        k1 = manim.ImageMobject(kernel1_img)
        k2 = manim.ImageMobject(kernel2_img)
        k3 = manim.ImageMobject(kernel3_img)
        k4 = manim.ImageMobject(kernel4_img)

        for k in (k1, k2, k3, k4):
            k.height = 1.5

        l1 = manim.Text("Kernel1\n(b=4)", **font_style).scale(0.5).set_color(manim.PINK)
        l2 = manim.Text("kernel2\n(b=1)", **font_style).scale(0.5).set_color(manim.GREEN)
        l3 = manim.Text("Kernel3\n(b=3)", **font_style).scale(0.5).set_color(manim.BLUE)
        l4 = manim.Text("kernel4\n(b=6)", **font_style).scale(0.5).set_color(manim.GRAY_B)

        kg1 = manim.Group(k1, l1).arrange(manim.DOWN, buff=0.15)
        kg2 = manim.Group(k2, l2).arrange(manim.DOWN, buff=0.15)
        kg3 = manim.Group(k3, l3).arrange(manim.DOWN, buff=0.15)
        kg4 = manim.Group(k4, l4).arrange(manim.DOWN, buff=0.15)

        kernels_g = manim.Group(kg1, kg2, kg3, kg4).arrange(manim.RIGHT, buff=0.7)
        kernels_g.to_edge(manim.UP).shift(manim.DOWN * 0.2)
        self.play(manim.FadeIn(kernels_g), runtime=0.6)
        self.wait(0.2)

        # --- Feature maps (4) ---
        fm1 = manim.ImageMobject(kernel_to_image(conv1, colormap='Oranges'))
        fm2 = manim.ImageMobject(kernel_to_image(conv2, colormap='Greens'))
        fm3 = manim.ImageMobject(kernel_to_image(conv3, colormap='Blues'))
        fm4 = manim.ImageMobject(kernel_to_image(conv4, colormap='Greys'))

        for fm in (fm1, fm2, fm3, fm4):
            fm.height = 1.5

        fl1 = manim.Text("F1", **font_style).scale(0.5).set_color(manim.PINK)
        fl2 = manim.Text("F2", **font_style).scale(0.5).set_color(manim.GREEN)
        fl3 = manim.Text("F3", **font_style).scale(0.5).set_color(manim.BLUE)
        fl4 = manim.Text("F4", **font_style).scale(0.5).set_color(manim.GRAY_B)

        fg1 = manim.Group(fm1, fl1).arrange(manim.DOWN, buff=0.1)
        fg2 = manim.Group(fm2, fl2).arrange(manim.DOWN, buff=0.1)
        fg3 = manim.Group(fm3, fl3).arrange(manim.DOWN, buff=0.1)
        fg4 = manim.Group(fm4, fl4).arrange(manim.DOWN, buff=0.1)

        features_g = manim.Group(fg1, fg2, fg3, fg4).arrange_in_grid(rows=1, cols=4, buff=0.35)
        # Place features in the center (between left input and right output)
        # features_g.move_to(manim.ORIGIN).shift(manim.DOWN * 1.0)
        features_g.to_edge(manim.DOWN).shift(manim.UP*0.2)
        self.play(manim.FadeIn(features_g), runtime=0.6)
        self.wait(0.3)

        # --- Pixel Shuffle (×2) of the 4 feature maps ---
        H, W = conv1.shape
        feats = np.stack([conv1, conv2, conv3, conv4], axis=0)             # (4,H,W)
        feats_t = torch.from_numpy(feats).float()[None, ...]               # (1,4,H,W)
        shuffler = nn.PixelShuffle(2)
        up = shuffler(feats_t).numpy()[0, 0]                               # (2H,2W)
        # up= gaussian_filter(up, sigma=10)
        up = up + np.random.normal(-0.8,0.8,up.shape)

        up_img = kernel_to_image(up, colormap='magma')
        up_m = manim.ImageMobject(up_img)
        up_m.height = 2

        up_label = manim.Text("Uniformly Noisy\nPixel Shuffled Output", **font_style).scale(0.6)
        up_group = manim.Group(up_m, up_label).arrange(manim.DOWN, buff=0.15)

        # Place the upscaled output to the RIGHT of the features
        # up_group.next_to(features_g, manim.RIGHT, buff=0.8)
        up_group.to_edge(manim.RIGHT).shift(manim.LEFT*0.2)
        self.play(manim.FadeIn(up_group), runtime=0.6)

        # Optional shape labels (kept simple, adjust as you like)
        shape_in = manim.Text(f"[4, {H}, {W}] features", **font_style2)
        shape_out = manim.Text(f"[1, {2*H}, {2*W}]", **font_style2)
        shape_in.next_to(features_g, manim.UP, buff=0.1)
        shape_out.next_to(up_group, manim.UP, buff=0.1)
        self.play(manim.FadeIn(shape_in), manim.FadeIn(shape_out), runtime=0.4)

        self.wait(2)


if __name__ == "__main__":
    os.system("manim -qh --resolution 1920,1080 shuffling_problem.py ShufflingProblem")
    # os.system("manim -ql --resolution 1920,1080 shuffling_problem.py ShufflingProblem")
