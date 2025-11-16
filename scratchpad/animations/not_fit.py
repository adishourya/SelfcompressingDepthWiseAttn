import manim

# white background
manim.config.background_color = manim.WHITE

happy_computer_path = "./computer-happy.svg"
sad_computer_path = "./computer-sad.svg"
ai_block_path = "./ai_block.svg"


class NOT_FIT(manim.Scene):
    def construct(self):
        # --- Load SVG images ---
        happy = manim.SVGMobject(happy_computer_path).scale(2)
        sad = manim.SVGMobject(sad_computer_path).scale(2)
        ai_block = manim.SVGMobject(ai_block_path).scale(1.0)

        # Start with the happy computer in the center
        self.play(manim.FadeIn(happy, run_time=0.1))

        # Positions for AI attempts: left, right, top, bottom
        positions = [
            manim.LEFT * 4,
            manim.RIGHT * 4,
            manim.UP * 3,
        ]

        # Amount to "enter" into the computer (10% of width/height)
        enter_offset = happy.width * 0.01

        for pos in positions:
            ai_block.move_to(pos)
            self.play(
                manim.FadeIn(ai_block, shift=pos/4, run_time=0.4)
            )

            # Move AI into computer slightly
            target = happy.get_center()
            self.play(
                ai_block.animate.move_to(target),
                run_time=0.3
            )

            # Wiggle / bounce
            self.play(
                manim.Wiggle(ai_block, scale_value=0.45, rotation_angle=0.5),
                run_time=0.5
            )

            # Retreat
            self.play(
                ai_block.animate.move_to(pos),
                run_time=0.3
            )

            self.play(manim.FadeOut(ai_block, run_time=0.2))

        # AI finally leaves (optional)
        # happy computer transforms into sad
        self.play(
            manim.Transform(happy, sad),
            run_time=1
        )

        self.wait(3)


if __name__ == "__main__":
    import os
    os.system("manim -qh --resolution 1920,1080 not_fit.py NOT_FIT")
