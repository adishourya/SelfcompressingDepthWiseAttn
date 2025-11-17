from manim import *

font_style0 = dict(font_size=10, font="Simple Nerd Font", color=GRAY)
font_style = dict(font_size=20, font="Simple Nerd Font", color=BLACK)

class DualLinePlots(Scene):
    def construct(self):
        # White background
        self.camera.background_color = WHITE

        # Data for left plot (Active Modules)
        left_points = [(0,400), (250, 398), (500, 395), (750, 385), (1000, 382)]
        
        # Data for right plot (Active Heads)
        right_points = [(0,64),(250, 64), (500, 64), (750, 62), (1000, 50)]

        # Create left axes (Active Modules)
        left_axes = NumberPlane(
            x_range=[0, 1000, 250],
            y_range=[350, 450, 25],
            x_length=5,
            y_length=5,
            axis_config={"color": BLACK, "stroke_width": 2},
            x_axis_config={"include_numbers": True, "numbers_to_include":[250, 500, 750, 1000]},
            y_axis_config={"include_numbers": True, "numbers_to_include":[350, 375, 400, 425, 450]},
        )
        

        # Create right axes (Active Heads)
        right_axes = NumberPlane(
            x_range=[0, 1000, 250],
            y_range=[45, 70, 5],
            x_length=5,
            y_length=5,
            axis_config={"color": BLACK, "stroke_width": 2},
            x_axis_config={"include_numbers": True, "numbers_to_include":[250, 500, 750, 1000]},
            y_axis_config={"include_numbers": True, "numbers_to_include":[40, 50, 60, 70]},
        )

        

        # Position axes side by side
        left_axes.shift(LEFT * 3.5)
        right_axes.shift(RIGHT * 3.5)

        # Set axis numbers color
        for axis in [left_axes, right_axes]:
            axis.get_x_axis().numbers.set_color(BLACK)
            axis.get_y_axis().numbers.set_color(BLACK)


        left_axes_labels = left_axes.get_axis_labels(
            x_label=Text("Epochs", **font_style),
            y_label=Text("Active Conv Kernels", **font_style)
        )
        
        right_axes_labels = right_axes.get_axis_labels(
            x_label=Text("Epochs", **font_style),
            y_label=Text("Active Attn Heads", **font_style)
        )

        
        # Add axes and labels
        #
        # 
        # self.play(Create(left_axes), Write(left_axes_labels))
        # self.play(Create(right_axes), Write(right_axes_labels))

        self.play(Create(left_axes), Write(left_axes_labels), Write(right_axes), Write(right_axes_labels))

        # Plot left graph (Active Modules)
        left_dots = VGroup()
        for x, y in left_points:
            dot = Dot(left_axes.coords_to_point(x, y), radius=0.08, color=ORANGE)
            left_dots.add(dot)
        
        left_lines = VGroup()
        for i in range(len(left_points)-1):
            line = Line(
                left_axes.coords_to_point(left_points[i][0], left_points[i][1]),
                left_axes.coords_to_point(left_points[i+1][0], left_points[i+1][1]),
                color=ORANGE,
                stroke_width=3
            )
            left_lines.add(line)

        # Plot right graph (Active Heads)
        right_dots = VGroup()
        for x, y in right_points:
            dot = Dot(right_axes.coords_to_point(x, y), radius=0.08, color=GREEN)
            right_dots.add(dot)
        
        right_lines = VGroup()
        for i in range(len(right_points)-1):
            line = Line(
                right_axes.coords_to_point(right_points[i][0], right_points[i][1]),
                right_axes.coords_to_point(right_points[i+1][0], right_points[i+1][1]),
                color=GREEN,
                stroke_width=3
            )
            right_lines.add(line)

        # Animate left plot
        self.play(Create(left_dots), Create(left_lines))
        # self.play(Create(left_lines))
        
        # Animate right plot
        self.play(Create(right_dots), Create(right_lines))
        # self.play(Create(right_lines))

        # Add titles
        # left_title = Text("(a) Pruning of modules", font_size=16, color=BLACK)
        # left_title.next_to(left_axes, UP, buff=0.3)
        
        # right_title = Text("(b) Pruning of attention heads", font_size=16, color=BLACK)
        # right_title.next_to(right_axes, UP, buff=0.3)

        # self.play(Write(left_title), Write(right_title))
        title = Text("Pruning In CUB2011 1.98M Model", font_size=12, font="Simple Nerd Font",color=GRAY_E)
        title.to_edge(UP)
        self.play(Write(title))

        self.wait(2)

if __name__ == "__main__":
    import os
    os.system("manim -qh --resolution 1920,1080 dual_plots.py DualLinePlots")
