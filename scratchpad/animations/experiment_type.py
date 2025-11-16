import manim

font_style = dict(font_size=100, font="Simple Nerd Font",color=manim.BLACK)

class TYPE_EXPERIMENT(manim.Scene):
    def construct(self):
        self.camera.background_color = manim.WHITE
        text = manim.Text("Experiment\nSettings",**font_style)
        self.play( manim.Write(text),run_time=3)
        self.wait(4)

if __name__ == "__main__":
    import os
    os.system("manim -qh --resolution 1000,600 experiment_type.py TYPE_EXPERIMENT")
