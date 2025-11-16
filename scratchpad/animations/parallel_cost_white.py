from manim import *

font_style = dict(font_size=24, font="Simple Nerd Font", color=BLACK)
font_style1 = dict(font_size=24, font="Simple Nerd Font")

class ParallelReduction(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        # Title
        title = Text("Parallel global reduction is logarithmic", **font_style)
        self.play(FadeIn(title, shift=UP*0.5), run_time=0.6)
        self.play(title.animate.to_edge(UP), run_time=0.6)

        # Vector shown briefly
        vector = list(range(1, 13))
        vec_text = Text(f"Vector = {vector}", **font_style)
        vec_text.next_to(title, DOWN, buff=1)
        self.play(FadeIn(vec_text, shift=UP*0.3), run_time=0.6)
        self.play(FadeOut(vec_text), run_time=0.4)

        # ===== Create 3 blocks =====
        block_positions = [LEFT*4, ORIGIN, RIGHT*4]
        blocks = VGroup()

        for i in range(3):
            rect = Rectangle(width=4, height=2, color=GREEN, fill_opacity=0.05)
            rect.move_to(block_positions[i])
            label = Text(f"SBlock {i+1}", **font_style).next_to(rect, UP, buff=0.2)
            blocks.add(VGroup(rect, label))

        self.play(LaggedStart(*[Create(b) for b in blocks], lag_ratio=0.15), run_time=0.8)

        # ===== Fill with 4-element chunks =====
        block_contents = VGroup()
        block_size = 4

        for i in range(3):
            chunk = vector[i*block_size:(i+1)*block_size]
            text = Text(f"{chunk}", **font_style)
            text.move_to(blocks[i][0].get_center())
            block_contents.add(text)

        self.play(LaggedStart(*[FadeIn(t) for t in block_contents], lag_ratio=0.15), run_time=0.8)

        # ------------------------------------------------------------
        # STEP 1 — PARALLEL BLOCK LOCAL REDUCTION (All blink together)
        # ------------------------------------------------------------
        step1 = Text("Step 1: Within Block Reductions (Fast)", **font_style1, color=BLUE)
        step1.next_to(blocks, DOWN, buff=0.2)
        self.play(FadeIn(step1), run_time=0.5)

        # Blink effect for parallel work
        blink_anims = []
        for i in range(3):
            rect = blocks[i][0]
            blink_anims.append(rect.animate.set_fill(BLUE, opacity=0.3))

        self.play(*blink_anims, run_time=0.3)
        self.play(*[rect[0].animate.set_fill(GREEN, opacity=0.05) for rect in blocks], run_time=0.3)

        # Compute and show block sums
        block_sums = []
        sum_texts = VGroup()
        for i in range(3):
            s = sum(vector[i*block_size:(i+1)*block_size])
            block_sums.append(s)

            t = Text(str(s), **font_style1, color=BLUE)
            t.move_to(blocks[i][0].get_center() + UP*0.7)
            sum_texts.add(t)

        self.play(LaggedStart(*[FadeIn(t, shift=UP*0.2) for t in sum_texts], lag_ratio=0.15), run_time=0.8)

        # ------------------------------------------------------------
        # STEP 2 — ACROSS BLOCK REDUCTION (Sequential, blink one by one)
        # ------------------------------------------------------------
        step2 = Text("Step 2: Across Block Reduction (SLOW)", **font_style1, color=RED)
        step2.next_to(step1, DOWN, buff=0.2)
        self.play(FadeIn(step2), run_time=0.5)

        # Blink block 1 and 2 one-by-one
        for i in [0, 1]:
            rect = blocks[i][0]
            self.play(rect.animate.set_fill(RED, opacity=0.3), run_time=0.25)
            self.play(rect.animate.set_fill(GREEN, opacity=0.05), run_time=0.25)

        # Combine block 1 + block 2
        combine12 = block_sums[0] + block_sums[1]
        combine12_text = Text(str(combine12), **font_style1, color=RED)
        combine12_text.move_to((blocks[0][0].get_center() + blocks[1][0].get_center())/2 + UP*1.0)

        self.play(FadeIn(combine12_text, shift=DOWN*0.2), run_time=0.5)

        # Blink block 3
        rect = blocks[2][0]
        self.play(rect.animate.set_fill(RED, opacity=0.3), run_time=0.25)
        self.play(rect.animate.set_fill(GREEN, opacity=0.05), run_time=0.25)

        final_sum = combine12 + block_sums[2]
        final_text = Text(f"Final Sum = {final_sum}", **font_style1, color=RED)
        final_text.next_to(title, DOWN)

        self.play(FadeIn(final_text), run_time=0.6)

        self.wait(1)

if __name__ == "__main__":
    import os
    os.system("manim -qh --resolution 1920,1080 parallel_cost_white.py ParallelReduction")
