import os
from manim import *
import numpy as np

AXIS_COLOR= BLACK
AXIS_NUM_COLOR = BLACK

AXIS_CONFIG = {
    "axis_config": {
        "stroke_color": AXIS_COLOR,
    },
    "x_axis_config": {
        "stroke_color": AXIS_COLOR,
        "include_numbers": True,
        "font_size": 18,
        "numbers_to_include": None,
        "number_config": {"color": AXIS_NUM_COLOR},   # <-- THIS IS WHAT YOU WERE MISSING
    },
    "y_axis_config": {
        "stroke_color": AXIS_COLOR,
        "include_numbers": True,
        "font_size": 18,
        "numbers_to_include": None,
        "number_config": {"color": AXIS_NUM_COLOR},   # <-- FIX
    },
    "tips": True,
}


def qbits(x, b=6.0, e=-8.0):
    b = b if b > 0 else 0
    x_scaled = x / np.exp2(e)
    x_clipped = np.clip(x_scaled, -1 * np.exp2(b - 1), np.exp2(b - 1) - 1)
    x_round = np.round(x_clipped)
    result = np.exp2(e) * x_round
    return (result, x_scaled, x_clipped, x_round), (-1 * np.exp2(b-1), np.exp2(b-1)-1)


class QuantizationAnimation(MovingCameraScene):
    def construct(self):

        # White background
        self.camera.background_color = WHITE

        # ====== AXIS COLOR FIX ======
        AXIS_COLOR = BLACK
        AXIS_NUM_COLOR = BLACK
        AXIS_CONFIG = {
            "axis_config": {"stroke_color": AXIS_COLOR},
            "x_axis_config": {
                "stroke_color": AXIS_COLOR,
                "color": AXIS_NUM_COLOR,
                "include_numbers": True,
                "font_size": 18,
            },
            "y_axis_config": {
                "stroke_color": AXIS_COLOR,
                "color": AXIS_NUM_COLOR,
                "include_numbers": True,
                "font_size": 18,
            },
            "tips": True,
        }
        # ============================

        # dont change this...
        e_bit = -3.5
        b_bit = 3.5
        x_input = np.linspace(-2, 2, 100)
        y_init = x_input

        (y_result, y1, y2, y3), (cl_l, cl_r) = qbits(x_input, b_bit, e_bit)

        # ============= AXES 1 =============
        ax1 = Axes(
            x_range=[x_input[0], x_input[-1], 5],
            y_range=[y_init[0], y_init[-1], 5],
            x_length=5,
            y_length=5,
            **AXIS_CONFIG
        )
        ax1_copy = ax1.copy()

        graph1 = ax1.plot_line_graph(x_input, y_init, line_color=ORANGE, vertex_dot_radius=0)
        graph1_title = Text("Linspaced Input (-3,3)", font_size=20, color=ORANGE)
        graph1_title.next_to(graph1, LEFT)


        # ============= AXES 2 (Scale) =============
        ax2 = Axes(
            x_range=[x_input[0], x_input[-1]],
            y_range=[y1[0], y1[-1]],
            x_length=5,
            y_length=7,
            **AXIS_CONFIG
        )
        graph2 = ax2.plot_line_graph(x_input, y1, line_color=RED, vertex_dot_radius=0)
        graph2_title = MathTex(r"\text{Scale } \frac{1}{2^e}", color=RED).scale(0.7)
        graph2_title.next_to(graph2, LEFT)

        # ============= AXES 3 (Clip) =============
        ax3 = Axes(
            x_range=[x_input[0], x_input[-1]],
            y_range=[y1[0], y1[-1]],
            x_length=5,
            y_length=7,
            **AXIS_CONFIG
        )
        graph3 = ax3.plot_line_graph(x_input, y2, line_color=YELLOW, vertex_dot_radius=0)
        graph3_title = MathTex(r"\text{clip }(-2^{b-1},\;2^{b-1}-1)", color=YELLOW).scale(0.7)
        graph3_title.next_to(graph3, LEFT)

        # ============= AXES 4 (Quantize) =============
        ax4 = Axes(
            x_range=[x_input[0], x_input[-1]],
            y_range=[y1[0], y1[-1]],
            x_length=5,
            y_length=7,
            **AXIS_CONFIG
        )
        graph4 = ax4.plot_line_graph(x_input, y3, line_color=GREEN, vertex_dot_radius=0)
        graph4_title = Text("Quantize torch.round\n(STE)", font_size=20, color=GREEN)
        graph4_title.next_to(graph4, LEFT)

        # ============= AXES 5 (Rescale back) =============
        ax5 = Axes(
            x_range=[x_input[0], x_input[-1]],
            y_range=[y_result[0], y_result[-1]],
            x_length=5,
            y_length=5,
            **AXIS_CONFIG
        )
        graph5 = ax5.plot_line_graph(x_input, y_result, line_color=PURPLE, vertex_dot_radius=0)
        graph5_title = MathTex(r"\text{Scale Down } 2^e", color=PURPLE).scale(0.7)
        graph5_title.next_to(graph5, LEFT)

        # ============= NUMBER PLANE FIX =============
        plane1 = NumberPlane(
            x_range=(x_input[0], x_input[-1]),
            y_range=(y_init[0], y_init[-1]),
            background_line_style={"stroke_width": 1, "stroke_color": GRAY},
            axis_config={"stroke_color": AXIS_COLOR},
            x_axis_config={"color": AXIS_NUM_COLOR},
            y_axis_config={"color": AXIS_NUM_COLOR},
        )

        # ============= Animation =============
        # self.play(Create(ax1), Create(graph1), Write(graph1_title))
        self.wait(0.5)

        # Scale
        self.play(
            Transform(ax1, ax2),
            ScaleInPlace(plane1, np.exp2(e_bit)),
            Transform(graph1, graph2),
            # Transform(graph1_title, graph2_title),
            run_time=1.2
        )

        # Clip phase
        clip_line1 = ax2.plot(lambda x: cl_l, color=YELLOW)
        clip_line2 = ax2.plot(lambda x: cl_r, color=YELLOW)

        # self.play(Transform(graph1_title, graph3_title))
        self.play(Create(clip_line1), Create(clip_line2), run_time=0.5)
        self.play(Uncreate(clip_line1), Uncreate(clip_line2), run_time=0.5)

        self.play(Transform(graph1, graph3))

        # Quantize + zoom-in
        # self.play(Transform(graph1_title, graph4_title))
        self.play(self.camera.frame.animate.scale(0.1).move_to(ORIGIN), run_time=1)
        self.play(Transform(graph1, graph4))
        self.play(self.camera.frame.animate.scale(10).move_to(ORIGIN), run_time=1)

        # Rescale back
        self.play(
            Transform(ax1, ax1_copy),
            ScaleInPlace(plane1, 1/np.exp2(e_bit)),
            Transform(graph1, graph5),
            # Transform(graph1_title, graph5_title),
        )
        self.wait(3)


if __name__ == "__main__":
    os.system("manim -qh --resolution 1000,1000 quant_animation_white.py QuantizationAnimation")
