
import os
from tkinter import RIGHT
import manim
import numpy as np

def qbits(x, b: float = 6.0, e: float = -8.0):
    b = b if b > 0 else 0
    x_scaled = x / np.exp2(e)
    x_clipped = np.clip(x_scaled, -1 * np.exp2(b - 1), np.exp2(b - 1) - 1)
    x_round = np.round(x_clipped)
    result = np.exp2(e) * x_round
    return result

class toSparse(manim.MovingCameraScene):
    def construct(self):
        ax = manim.Axes(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=6,
            y_length=6,
            axis_config={"include_tip": False}
        )

        # Labels for axes
        ax_labels = ax.get_axis_labels(x_label="x", y_label="q(x)")

        plane=manim.NumberPlane(x_range=(-2,2),
                           y_range=(-2,2),
                           background_line_style={"stroke_width": 1, "stroke_color": manim.GRAY},)

        # ValueTrackers for b and e
        b_tracker = manim.ValueTracker(3.5)
        e_tracker = manim.ValueTracker(-3.5)

        # Initial plot
        def get_quant_plot():
            x_vals = np.linspace(-2, 2, 100)
            y_vals = qbits(x_vals, b_tracker.get_value(), e_tracker.get_value())
            return ax.plot_line_graph(x_vals, y_vals, line_color=manim.PURPLE,vertex_dot_radius=0)

        quant_graph = get_quant_plot()

        # b and e value displays
        b_display = manim.DecimalNumber(b_tracker.get_value(), num_decimal_places=2).next_to(ax, manim.LEFT).shift(manim.UP * 0.5)
        b_label = manim.Text("b =", font="Simple Nerd Font", font_size=24).next_to(b_display, manim.LEFT)

        e_display = manim.DecimalNumber(e_tracker.get_value(), num_decimal_places=2).next_to(ax, manim.LEFT).shift(manim.DOWN * 0.5)
        e_label = manim.Text("e =", font="Simple Nerd Font", font_size=24).next_to(e_display, manim.LEFT)

        # titles
        title1 = manim.MarkupText(
        f"""
        Bit Depth: <span fgcolor="{manim.YELLOW}">Normal</span>
        Exp Depth: <span fgcolor="{manim.YELLOW}">Normal</span>
        Improves:  <span fgcolor="{manim.YELLOW}">Normal</span>
        TradeOff:  <span fgcolor="{manim.YELLOW}">Normal</span>
        """, color=manim.WHITE,font_size=15,font="Simple Nerd Font",
        ).next_to(ax,manim.RIGHT).shift(manim.RIGHT)

        title2 = manim.MarkupText(
        f"""
        Bit Depth: <span fgcolor="{manim.YELLOW}">Normal</span>
        Exp Depth: <span fgcolor="{manim.RED}">HIGH</span>
        Improves:  <span fgcolor="{manim.YELLOW}">Normal</span>
        TradeOff:  <span fgcolor="{manim.YELLOW}">Coverage Collapse</span>
        """, color=manim.WHITE,font_size=15,font="Simple Nerd Font",
        ).next_to(ax,manim.RIGHT).shift(manim.RIGHT)

        title3 = manim.MarkupText(
        f"""
        Bit Depth: <span fgcolor="{manim.RED}">High</span>
        Exp Depth: <span fgcolor="{manim.YELLOW}">Normal</span>
        Improves:  <span fgcolor="{manim.GREEN}">More Coverage</span>
        TradeOff:  <span fgcolor="{manim.YELLOW}">Normal</span>
        """, color=manim.WHITE,font_size=15,font="Simple Nerd Font",
        ).next_to(ax,manim.RIGHT).shift(manim.RIGHT)

        title4 = manim.MarkupText(
        f"""
        Bit Depth: <span fgcolor="{manim.RED}">HIGH</span>
        Exp Depth: <span fgcolor="{manim.RED}">HIGH</span>
        Improves:  <span fgcolor="{manim.GREEN}">Precision</span>
        TradeOff:  <span fgcolor="{manim.RED}">Input Coverage</span>
        """, color=manim.WHITE,font_size=15,font="Simple Nerd Font",
        ).next_to(ax,manim.RIGHT).shift(manim.RIGHT)

        title5 = manim.MarkupText(
        f"""
        Bit Depth: <span fgcolor="{manim.RED}">HIGH</span>
        Exp Depth: <span fgcolor="{manim.GREEN}">LOW</span>
        Improves:  <span fgcolor="{manim.GREEN}">Storage</span>
        TradeOff:  <span fgcolor="{manim.RED}">Less Precise</span>
        """, color=manim.WHITE,font_size=15,font="Simple Nerd Font",
        ).next_to(ax,manim.RIGHT).shift(manim.RIGHT)

        title6 = manim.MarkupText(
        f"""
        Bit Depth: <span fgcolor="{manim.GREEN}">LOW</span>
        Exp Depth: <span fgcolor="{manim.GREEN}">LOW</span>
        Improves:  <span fgcolor="{manim.GREEN}">Storage</span>
        TradeOff:  <span fgcolor="{manim.RED}">Input Coverage</span>
        """, color=manim.WHITE,font_size=15,font="Simple Nerd Font",
        ).next_to(ax,manim.RIGHT).shift(manim.RIGHT)

        title7 = manim.MarkupText(
        f"""
        Bit Depth: <span fgcolor="{manim.YELLOW}">0</span>
        Exp Depth: <span fgcolor="{manim.YELLOW}">Any</span>
        Quant:     <span fgcolor="{manim.YELLOW}">Sparse</span>
        """, color=manim.WHITE,font_size=20,font="Simple Nerd Font",
        ).next_to(ax,manim.RIGHT).shift(manim.RIGHT)


        self.add(ax, ax_labels, quant_graph, b_display, b_label, e_display, e_label,plane,title1)

        def update_graph(mobj):
            mobj.become(get_quant_plot())

        def update_b(mobj):
            mobj.set_value(b_tracker.get_value())

        def update_e(mobj):
            mobj.set_value(e_tracker.get_value())

        quant_graph.add_updater(update_graph)
        b_display.add_updater(update_b)
        e_display.add_updater(update_e)

        # Animate: improve e (more precision)
        self.play(manim.Transform(title1,title2))
        self.wait(1)
        self.play(manim.Indicate(e_display))
        self.play(e_tracker.animate.set_value(-8.0), run_time=3)
        self.wait(2)
        self.play(e_tracker.animate.set_value(-3.5), run_time=3)

        # Animate: increase b (more precision)
        self.play(manim.Transform(title1,title3))
        self.wait(1)
        self.play(manim.Indicate(b_display))
        self.play(b_tracker.animate.set_value(6), run_time=3)
        self.wait(2)

        # Animate: improve e (more precision)
        self.play(manim.Transform(title1,title4))
        self.wait(1)
        self.play(manim.Indicate(e_display))
        self.play(e_tracker.animate.set_value(-6.0), run_time=3)
        self.wait(2)

        self.play(self.camera.frame.animate.scale(0.1).move_to(manim.ORIGIN), run_time=1)
        self.wait(1)
        self.play(self.camera.frame.animate.scale(10).move_to(manim.ORIGIN), run_time=1)
        self.wait(2)

        # Animate: decrease e (bad precision)
        self.play(manim.Transform(title1,title5))
        self.wait(1)
        self.play(manim.Indicate(e_display))
        self.play(e_tracker.animate.set_value(-1.0), run_time=3)
        self.wait(2)

        # Animate: decrease b (bad coverage)
        self.play(manim.Transform(title1,title6))
        self.wait(1)
        self.play(manim.Indicate(b_display))
        self.play(b_tracker.animate.set_value(1), run_time=3)
        self.wait(2)

        # Animate: set b to 0 (makes output sparse)
        self.play(manim.Transform(title1,title7))
        self.wait(1)
        self.play(manim.Indicate(b_display))
        self.play(b_tracker.animate.set_value(0), run_time=3)
        self.wait(2)

        # Animate: no place for mantissa
        self.play(manim.Indicate(e_display))
        self.wait(1)
        self.play(e_tracker.animate.set_value(-8.0), run_time=3)
        self.wait(2)

        quant_graph.clear_updaters()
        b_display.clear_updaters()
        e_display.clear_updaters()

        self.wait()

if __name__ == "__main__":
    # os.system("manim -qh --resolution 1920,1080 quant_sparse.py toSparse")
    os.system("manim -qh --resolution 960,1080 quant_sparse.py toSparse")
