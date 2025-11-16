import manim

font_style = dict(font_size=100, font="Simple Nerd Font",color=manim.BLACK)

class TYPE_HYPOTHESIS(manim.Scene):
    def construct(self):
        self.camera.background_color = manim.WHITE
        text = manim.Text("Hypothesis",**font_style)
        self.play( manim.Write(text),run_time=3)
        self.wait(4)

if __name__ == "__main__":
    import os
    os.system("manim -qh --resolution 1000,600 hypothesis_type.py TYPE_HYPOTHESIS")
