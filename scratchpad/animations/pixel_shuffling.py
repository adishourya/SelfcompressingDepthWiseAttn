import os
import torch
import torch.nn as nn
import manim

font_style = dict(font_size=22, font="Simple Nerd Font")

class PixelShuffling(manim.Scene):
    def construct(self):
        b = 2
        h, w = 3, 3
        channels = b * b  # b²

        # Title
        title = manim.Text("Pixel Shuffling", **font_style)
        title.to_edge(manim.UP)
        self.play(manim.Write(title), run_time=0.4)

        # Input tensor (dummy data for visualization)
        inp = torch.arange(channels * h * w).float().reshape(1, channels, h, w)

        # Colors for channels
        colors = manim.color_gradient(
            [manim.RED, manim.YELLOW, manim.GREEN, manim.BLUE], channels
        )

        # Create input visualization: each channel = h×w block
        in_group = manim.VGroup()
        for ch in range(channels):
            grid = manim.VGroup()
            for r in range(h):
                for c in range(w):
                    sq = manim.Square(
                        side_length=0.5,
                        fill_color=colors[ch],
                        fill_opacity=0.8,
                        stroke_width=1,
                    )
                    sq.move_to(manim.RIGHT * c * 0.5 + manim.UP * (-r * 0.5))
                    grid.add(sq)
            in_group.add(grid)

        # Arrange the channels into b×b layout (PixelShuffle input)
        in_group.arrange_in_grid(rows=b, cols=b, buff=0.5)
        in_group.move_to(manim.ORIGIN)

        # Shape text below (fixed y position)
        shape_text = manim.Text(f"[{b*b}, {h}, {w}] (B², H, W)", font_size=28)
        shape_text.move_to(manim.DOWN * 3)  # fixed position, won't shift

        # Show input
        self.play(
            manim.LaggedStart(*[manim.FadeIn(g) for g in in_group], lag_ratio=0.05),
            manim.FadeIn(shape_text),
            run_time=1
        )
        self.wait(0.5)

        # Pixel shuffle output
        shuffler = nn.PixelShuffle(b)
        _ = shuffler(inp)  # just to confirm correctness

        # Create output visualization [1, h*b, w*b]
        out_group = manim.VGroup()
        for r in range(h * b):
            for c in range(w * b):
                br, rr = divmod(r, h)
                bc, cc = divmod(c, w)
                ch_idx = br * b + bc
                sq = manim.Square(
                    side_length=0.5,
                    fill_color=colors[ch_idx],
                    fill_opacity=0.8,
                    stroke_width=1,
                )
                sq.move_to(manim.RIGHT * c * 0.5 + manim.UP * (-r * 0.5))
                out_group.add(sq)

        # Center the big grid on the same origin as the small one
        out_group.move_to(in_group.get_center())

        # Animate rearrangement
        anims = []
        for ch, sub_grid in enumerate(in_group):
            br = ch // b
            bc = ch % b
            for rr in range(h):
                for cc in range(w):
                    r_big = rr * b + br
                    c_big = cc * b + bc
                    idx_big = r_big * (w * b) + c_big
                    anims.append(sub_grid[rr * w + cc].animate.move_to(out_group[idx_big].get_center()))

        self.play(*anims, run_time=2)

        # Change shape text without moving position
        new_shape_text = manim.Text(f"[1, {h*b}, {w*b}] (1, H×B, W×B)", font_size=28)
        new_shape_text.move_to(shape_text.get_center())
        self.play(manim.Transform(shape_text, new_shape_text), run_time=0.5)

        self.wait(1)


if __name__ == "__main__":
    # os.system("manim -ql --resolution 1920,1080 pixel_shuffling.py PixelShuffling")
    os.system("manim -qh --resolution 1920,1080 pixel_shuffling.py PixelShuffling")
