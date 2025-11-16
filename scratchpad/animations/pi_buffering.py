import manim

font_style = dict(font_size=100, font="Simple Nerd Font",color=manim.BLACK)

# white background
manim.config.background_color = manim.WHITE


class PI_BUFFERING(manim.Scene):
    def construct(self):
        # The full pi digits we want to show
        pi_digits = "3.141592653589"
        
        # Start with "π = 3."
        pi_text = manim.Text("π=3.14", **font_style)
        self.add(pi_text)
        pi_text.to_edge(manim.LEFT)
        self.wait(0.5)

        # Create a VGroup for the next digits so we can use Write
        pi_group = manim.VGroup(*[manim.Text("", **font_style)])
        # We'll just append digits incrementally
        for digit in pi_digits[4:]:  # skip "3."
            # Create a Text for the new digit
            next_digit = manim.Text(digit, **font_style)
            next_digit.next_to(pi_text, manim.RIGHT, buff=0.05)
            
            # Blinking ellipses before revealing the digit
            ellipses = manim.Text(" ....", font_size=100, color=manim.RED)
            ellipses.next_to(pi_text, manim.RIGHT, buff=0.05)
            for _ in range(3):
                self.play(manim.Write(ellipses), run_time=0.3)
                self.play(manim.FadeOut(ellipses), run_time=0.3)
            
            # Write the new digit
            self.play(manim.Write(next_digit), run_time=0.4)
            
            # Shift the next digit’s position for subsequent digits
            pi_text = manim.VGroup(pi_text, next_digit)

        self.wait(1)


if __name__ == "__main__":
    import os
    os.system("manim -qh --resolution 1920,1080 pi_buffering.py PI_BUFFERING")
