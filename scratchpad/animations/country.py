from manim import *

photo1 = "./dataset_images/country2.jpg"
photo2 = "./dataset_images/country3.jpg"

font_style = dict(font_size=20, font="Simple Nerd Font",color=GRAY)

class BirdGrid(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        # Load images
        img1 = ImageMobject(photo1).scale(1.5)
        img2 = ImageMobject(photo2).scale(1.5)

        # Labels
        label1 = Text("Input 1" , **font_style)
        label2 = Text("Input 2", **font_style)

        # Position images manually for 2x2 grid
        img1.to_edge(UP+LEFT)
        label1.next_to(img1, DOWN, buff=0.2)

        img2.next_to(img1, RIGHT, buff=1)
        label2.next_to(img2, DOWN, buff=0.2)


        # Animate images and labels
        self.play(FadeIn(img1), Write(label1), runtime=0.1)
        self.play(FadeIn(img2), Write(label2),runtime=0.1)
        self.wait(1)

if __name__ == "__main__":
    import os
    os.system("manim -qh --resolution 1920,1000 country.py BirdGrid")
