
from manim import *

font_style = dict(font_size=15, font="Simple Nerd Font",color=BLACK)

class TransformerBlockDiagram(Scene):
    def construct(self):
        # Set white background
        self.camera.background_color = WHITE

        # Define colors
        yellow = "#FFEB84"
        green = "#84C49E"
        orange = "#FFA974"
        
        # Define the blocks
        blocks = [
            {"label": "Expand\nBlock(ConvD)", "color": yellow},
            {"label": "Upscaling\nBlock(ConvT)", "color": green},
            {"label": "Depthwise\nBlock(ConvD)", "color": orange},
            {"label": "Projection\nBlock(Conv)", "color": green},
            {"label": "Linear\nAttention", "color": yellow},
        ]
        
        # Create rectangles and labels
        rects = VGroup()
        for b in blocks:
            rect = RoundedRectangle(
                corner_radius=0.15,
                width=2,  # smaller width
                height=2, # smaller height
                fill_color=b["color"],
                fill_opacity=0.8,
                stroke_width=1
            )
            label = Text(b["label"], **font_style)  # black text for visibility
            rect_group = VGroup(rect, label)
            rect_group.label = label
            rects.add(rect_group)
        
        # Arrange in a row
        rects.arrange(RIGHT, buff=0.1)
        
        # Draw arrows between blocks
        arrows = VGroup()
        for i in range(len(rects)-1):
            arrow = Arrow(
                start=rects[i].get_right(),
                end=rects[i+1].get_left(),
                buff=0.1,
                stroke_width=5,
                color=BLACK
            )
            arrows.add(arrow)
        
        # Create a box around everything (Transformer Block)
        box = RoundedRectangle(
            width=rects.width + 1,
            height=rects.height + 0.8,
            corner_radius=0.3,
            color=GREY,
            stroke_width=2
        )
        box.move_to(rects.get_center())
        label_box = Text("Transformer Block from Efficient VIT", **font_style).next_to(box, DOWN, buff=0.2)
        
        # Add all elements to the scene
        self.play(Create(box))
        self.play(Create(rects), run_time=1.5)
        self.play(*[Create(arrow) for arrow in arrows])
        self.play(Write(label_box))
        self.wait(2)

        
if __name__ == "__main__":
    import os
    os.system("manim -qh --resolution 1920,1080 eff_vit_diag.py TransformerBlockDiagram")
