from manim import *

photo1 = "./dataset_images/Philadelphia_Vireo_0003_156565.jpg"
photo2 = "./dataset_images/Golden_Winged_Warbler_0030_164462.jpg"
photo3 = "./dataset_images/Hooded_Warbler_0008_164641.jpg"
photo4 = "./dataset_images/Kentucky_Warbler_0015_795867.jpg"

font_style = dict(font_size=20, font="Simple Nerd Font",color=GRAY)

class BirdGrid(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        # Load images
        img1 = ImageMobject(photo1).scale(1.2)
        img2 = ImageMobject(photo2).scale(1.2)
        img3 = ImageMobject(photo3).scale(1.2)
        img4 = ImageMobject(photo4).scale(2)

        # Labels
        label1 = Text("Philadelphia Vireo", **font_style)
        label2 = Text("Golden-winged Warbler", **font_style)
        label3 = Text("Hooded Warbler", **font_style)
        label4 = Text("Kentucky Warbler", **font_style)

        # Position images manually for 2x2 grid
        img1.to_edge(UP+LEFT)
        label1.next_to(img1, DOWN, buff=0.2)

        img2.next_to(img1, RIGHT, buff=1)
        label2.next_to(img2, DOWN, buff=0.2)

        img3.next_to(img1, DOWN, buff=1)
        label3.next_to(img3, DOWN, buff=0.2)

        img4.next_to(img3, RIGHT, buff=1)
        label4.next_to(img4, DOWN, buff=0.2)

        # Animate images and labels
        self.play(FadeIn(img1), Write(label1), runtime=0.1)
        self.play(FadeIn(img2), Write(label2),runtime=0.1)
        self.play(FadeIn(img3), Write(label3),runtime=0.1)
        self.play(FadeIn(img4), Write(label4),runtime=0.1)
        self.wait(1)

if __name__ == "__main__":
    import os
    os.system("manim -qh --resolution 1500,1500 cub_dataset.py BirdGrid")
