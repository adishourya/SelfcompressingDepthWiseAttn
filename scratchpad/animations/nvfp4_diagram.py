
from manim import *
font_style = dict(font_size=30, font="Simple Nerd Font",color=GRAY)
font_style2 = dict(font_size=48, font="Simple Nerd Font",color=BLACK)

class NVFP4Diagram(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        # --- Colors ---
        sign_color = GRAY
        exponent_color = GREEN
        mantissa_color = PURPLE

        # --- Squares (bit boxes) ---
        bit_size = 1.2

        sign_box = Square(side_length=bit_size, color=sign_color, fill_color=sign_color, fill_opacity=0.6)
        exp_box1 = Square(side_length=bit_size, color=exponent_color, fill_color=exponent_color, fill_opacity=0.6)
        exp_box2 = Square(side_length=bit_size, color=exponent_color, fill_color=exponent_color, fill_opacity=0.6)
        mant_box = Square(side_length=bit_size, color=mantissa_color, fill_color=mantissa_color, fill_opacity=0.6)

        # Arrange boxes in a row
        boxes = VGroup(sign_box, exp_box1, exp_box2, mant_box)
        boxes.arrange(RIGHT, buff=0.2)
        boxes.move_to(ORIGIN)

        # --- Labels ---
        sign_label = Text("S", **font_style2).move_to(sign_box)
        exp_label1 = Text("E", **font_style2).move_to(exp_box1)
        exp_label2 = Text("E", **font_style2).move_to(exp_box2)
        mant_label = Text("M", **font_style2).move_to(mant_box)

        labels = VGroup(sign_label, exp_label1, exp_label2, mant_label)

        # --- Animation Sequence ---
        # Show title
        title = Text("Micro Bit Designs", **font_style)
        title.to_edge(DOWN)
        self.play(FadeIn(title, shift=DOWN), run_time=0.7)
        self.wait(0.5)

        # Animate each box popping in
        for box, label in zip(boxes, labels):
            self.play(
                GrowFromCenter(box),
                run_time=0.4
            )
            self.play(
                Write(label),
                run_time=0.3
            )
            self.wait(0.1)

        self.wait(2)


if __name__ == "__main__":
    import os
    os.system("manim -qh --resolution 1920,1080 nvfp4_diagram.py NVFP4Diagram")
