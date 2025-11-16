from tkinter import BOTTOM
from manim import *

font_style0 = dict(font_size=10, font="Simple Nerd Font",color=GRAY)
font_style = dict(font_size=40, font="Simple Nerd Font",color=BLACK)

class RealLinePlot(Scene):
    def construct(self):
        # White background
        self.camera.background_color = WHITE


        # Points to plot
        points = [(32, 210), (64, 95),(96,94), (128, 94)]

        # Create axes with numbers
        axes = NumberPlane(
            x_range=[32, 128, 32],
            y_range=[50, 250, 50],
            x_length=8,
            y_length=8,
            axis_config={"color": BLACK, "include_numbers": True, "font_size":30},  # include numbers
            x_axis_config={"color": BLUE, "include_tip":False},
            y_axis_config={"color": BLUE, "include_tip": False},
        )
        axes_labels = axes.get_axis_labels(
            x_label=Text("Resolution", **font_style),
            y_label=Text("Convolution GB/s", **font_style)
        )

        x = axes.get_x_axis()
        x.numbers.set_color(BLACK)

        y = axes.get_y_axis()
        y.numbers.set_color(BLACK)

        self.add(axes, axes_labels)
        title = Text("As measured in NVIDIA 4070 Mobile",**font_style0)
        title.to_edge(DOWN)
        self.add(title)
        self.play(Write(axes),Write(axes_labels))



        # Add dots
        for x, y in points:
            dot = Dot(axes.coords_to_point(x, y), DEFAULT_DOT_RADIUS*2,color=ORANGE)
            self.add(dot)

        # Connect points with lines
        for i in range(len(points)-1):
            line = Line(
                axes.coords_to_point(points[i][0], points[i][1]),
                axes.coords_to_point(points[i+1][0], points[i+1][1]),
                color=BLACK
            )
            self.play(Write(line))


if __name__ == "__main__":
    import os
    os.system("manim -qh --resolution 1500,1500 throughput_conv.py RealLinePlot")
