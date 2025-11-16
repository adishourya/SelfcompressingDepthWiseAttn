
from manim import *

font_style = dict(font_size=30, font="Simple Nerd Font",color=BLACK)

class BitBarChart(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        chart = BarChart(values=[5.1,4.1,4.3,4.0,3.2,2.5,2.5,2.1,2.1],
                         y_range=[0, 6, 1],
                         bar_names=["Embed", "L1","L2","L3","L4","L5","L6","L7","L8"],
                     bar_colors=[ORANGE]*8)

        x = chart.get_x_axis()
        x.set_color(BLACK)

        y = chart.get_y_axis()
        y.set_color(BLACK)


        c_bar_lbls = chart.get_bar_labels(
            color=BLACK, label_constructor=MathTex, font_size=30
        )

        title= Text("Bit Depth Across Layers",**font_style)
        title.next_to(chart,DOWN)
        self.add(title)

        self.play(Write(c_bar_lbls))
        self.play(Write(chart))

if __name__ == "__main__":
    import os
    os.system("manim -qh --resolution 1920,1080 bar_chart.py BitBarChart")
