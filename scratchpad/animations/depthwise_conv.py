from manim import *
import numpy as np
import matplotlib.pyplot as plt

class ConvFilterSizeDemo(ThreeDScene):
    def construct(self):
        # Parameters
        k_h, k_w = 3, 3
        C_in = 6  # channels for demo

        # Load dolphin image and create fake feature stack
        img_path = "./dolphin.jpg"
        img = plt.imread(img_path).astype(float)
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:,:,:3]
        img = img / 255.0
        H, W, _ = img.shape

        # Fake features: just copy grayscale image across channels
        gray = np.mean(img, axis=2)
        features = np.stack([gray for _ in range(C_in)], axis=-1)  # H,W,C

        # Axes setup
        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)
        axes = ThreeDAxes(
            x_range=[0, W, 1],
            y_range=[0, H, 1],
            z_range=[0, C_in, 1],
            x_length=6, y_length=6, z_length=3
        )
        self.add(axes)

        # Feature volume as a cube mesh
        vol_box = Cube(side_length=1, fill_opacity=0.05).scale(3)
        vol_box.shift([3,3,1.5])
        self.add(vol_box)

        # Normal convolution prism (full depth)
        normal_kernel = Prism(
            dimensions=(k_w*(6/W), k_h*(6/H), C_in*(3/C_in)),
            fill_color=BLUE, fill_opacity=0.4,
            stroke_color=BLUE
        )

        # Depthwise convolution prism (depth=1 channel)
        depthwise_kernel = Prism(
            dimensions=(k_w*(6/W), k_h*(6/H), 1*(3/C_in)),
            fill_color=GREEN, fill_opacity=0.4,
            stroke_color=GREEN
        )

        # Positions
        normal_kernel.move_to([0.5, 0.5, 1.5])
        depthwise_kernel.move_to([0.5, 0.5, 0.5])

        # Labels
        normal_label = Text("Normal Conv\n(k_h × k_w × C_in)", font_size=28).next_to(normal_kernel, UP)
        depthwise_label = Text("Depthwise Conv\n(k_h × k_w × 1)", font_size=28).next_to(depthwise_kernel, DOWN)

        # Show
        self.play(FadeIn(normal_kernel), Write(normal_label))
        self.wait(1)
        self.play(ReplacementTransform(normal_kernel.copy(), depthwise_kernel), Write(depthwise_label))
        self.wait(2)

        # Animate sliding of kernels
        for x in np.linspace(0.5, 5.0, 5):
            self.play(normal_kernel.animate.move_to([x, 0.5, 1.5]), run_time=0.3)
        self.wait(0.5)
        for x in np.linspace(0.5, 5.0, 5):
            self.play(depthwise_kernel.animate.move_to([x, 0.5, 0.5]), run_time=0.3)

        self.wait(2)


if __name__ == "__main__":
    import os
    os.system("manim -ql --resolution 1920,1080 depthwise_conv.py ConvFilterSizeDemo")
