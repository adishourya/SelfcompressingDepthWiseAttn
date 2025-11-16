
import manim

# white background
manim.config.background_color = manim.WHITE


class LAYER_THROUGHPUT(manim.Scene):
    def construct(self):
        # --- Create boxes for layers ---
        num_layers = 5
        boxes = []
        spacing = 3  # horizontal spacing

        for i in range(num_layers):
            box = manim.Rectangle(width=2, height=2, color=manim.BLACK)
            box.move_to(manim.RIGHT * i * spacing)
            label = manim.Text(f"Layer {i+1}", font_size=24).next_to(box, manim.DOWN, buff=0.3)
            self.add(box, label)
            boxes.append(box)

        # --- Create the pi number ---
        pi = manim.MathTex(r"\pi", font_size=48, color=manim.BLUE)
        pi.move_to(boxes[0].get_top())  # start above first box

        self.add(pi)

        # --- Animate pi jumping from one box to the next ---
        for i in range(num_layers):
            # Move slowly above the box
            target = boxes[i].get_top() + manim.UP*0.5
            self.play(
                pi.animate.move_to(target),
                run_time=1.5
            )
            # Drop into the box slightly
            drop_target = boxes[i].get_center()
            self.play(
                pi.animate.move_to(drop_target),
                run_time=1
            )
            # small pause
            self.wait(0.3)

        # End with pi in the last box
        self.wait(1)


if __name__ == "__main__":
    import os
    os.system("manim -qh --resolution 1920,1080 layer_throughput.py LAYER_THROUGHPUT")
