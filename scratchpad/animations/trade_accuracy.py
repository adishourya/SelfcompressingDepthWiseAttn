from manim import *

font_style = dict(font_size=100, font="Simple Nerd Font",color=BLACK)

class LowBitClipping(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        # Full precision pi as text
        pi_full = Text("3.14159265358979", **font_style)
        pi_full.to_edge(LEFT)

        # π = on the left
        pi_text = Text("π=", **font_style)
        pi_text.next_to(pi_full, LEFT, buff=0.1)

        pi_group = VGroup(pi_text, pi_full)
        pi_group.move_to(ORIGIN)

        self.play(Write(pi_group))
        self.wait(1)

        # Low-bit version (Text, not DecimalNumber)
        pi_lowbit = Text("3.14159", **font_style)
        pi_lowbit.next_to(pi_text, RIGHT, buff=0.1)

        # Transform using text → no white text bug
        self.play(Wiggle(pi_full))
        self.play(Transform(pi_full, pi_lowbit))
        self.wait(2)


if __name__ == "__main__":
    import os
    os.system("manim -qh --resolution 1920,1080 trade_accuracy.py LowBitClipping")
