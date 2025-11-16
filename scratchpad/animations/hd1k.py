from manim import *

photo1 = "./dataset_images/hd1k_frames.png"

font_style = dict(font_size=20, font="Simple Nerd Font",color=GRAY)

class BirdGrid(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        # Load images
        img1 = ImageMobject(photo1).scale(1.2)

        # Labels
        label1 = Text("Flow Frames" , **font_style)

        # Position images manually for 2x2 grid
        img1.to_edge(UP+LEFT)
        label1.next_to(img1, DOWN, buff=0.2)

        # Animate images and labels
        self.play(FadeIn(img1), Write(label1), runtime=0.1)
        self.wait(1)

if __name__ == "__main__":
    import os
    os.system("manim -qh --resolution 1000,1000 hd1k.py BirdGrid")
