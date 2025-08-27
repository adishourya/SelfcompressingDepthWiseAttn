from manim import *
import numpy as np

font_style = dict(font_size=16, font="Simple Nerd Font")
class VectorOpsComparison(Scene):
    def construct(self):
        # Title
        title = Text("Parallel Cost of Map and Reduction Ops", **font_style)
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.to_edge(UP))
        self.wait(1)
        self.play(FadeOut(title))

        # Setup
        n = 12  # Total elements
        block_size = 4  # Elements per shared memory block
        num_blocks = 3  # Number of shared memory blocks

        # Create vectors
        vector_a = [i+1 for i in range(n)]
        vector_b = [2*i for i in range(n)]

        # Display vectors
        vector_a_text = Text("Vector A: " + str(vector_a), **font_style)
        vector_b_text = Text("Vector B: " + str(vector_b), **font_style)
        vector_group = VGroup(vector_a_text, vector_b_text).arrange(DOWN, aligned_edge=LEFT)
       
        self.play(Write(vector_a_text), Write(vector_b_text))
        self.wait(1)

        # Show shared memory blocks
        blocks_title = Text("Suppose Shared Memory Blocks can handle only 4 elements each", **font_style)
        blocks_title.to_edge(UP)
        # blocks_title.next_to(vector_group, DOWN, aligned_edge=LEFT).shift(DOWN*0.5)
        self.play(Write(blocks_title))
        self.play(FadeOut(vector_a_text),FadeOut(vector_b_text))
        
        # Create blocks
        blocks = VGroup()
        block_positions = [LEFT*5, ORIGIN, RIGHT*5]
        for i in range(num_blocks):
            block = Rectangle(height=2,width=4,color=BLUE,fill_opacity=0.2)
            # block = Square(side_length=3, color=BLUE, fill_opacity=0.2)
            block.move_to(block_positions[i])
            block_label = Text(f"Block {i+1}", **font_style).next_to(block, UP)
            blocks.add(VGroup(block, block_label))
        
        self.play(Create(blocks))
        self.wait(1)

        # Show data distribution to blocks
        data_texts = VGroup()
        for i in range(num_blocks):
            start_idx = i * block_size
            end_idx = start_idx + block_size
            a_segment = vector_a[start_idx:end_idx]
            b_segment = vector_b[start_idx:end_idx]
            
            text = Text(f"A[{start_idx}:{end_idx-1}]: {a_segment}\nB[{start_idx}:{end_idx-1}]: {b_segment}", 
                       **font_style)
            text.move_to(blocks[i].get_center())
            data_texts.add(text)
        
        self.play(Write(data_texts))
        self.wait(1)

        # Vector Addition - O(1)
        addition_title = Text("Vector Addition: O(1) - All elements processed in parallel", 
                             **font_style, color=GREEN)
        addition_title.to_edge(DOWN)
        self.play(Write(addition_title))
        self.wait(1)

        # Show addition happening simultaneously in all blocks
        addition_animations = []
        result_texts = VGroup()
        
        for i in range(num_blocks):
            start_idx = i * block_size
            end_idx = start_idx + block_size
            a_segment = vector_a[start_idx:end_idx]
            b_segment = vector_b[start_idx:end_idx]
            result = [a + b for a, b in zip(a_segment, b_segment)]
            
            result_text = Text(f"Result: {result}", **font_style, color=GREEN)
            result_text.move_to(blocks[i].get_center() + DOWN*0.8)
            result_texts.add(result_text)
            
            # Create highlight animation for each block
            addition_animations.append(Indicate(blocks[i], color=GREEN))
        
        # Play all animations simultaneously to show parallel processing
        self.play(*addition_animations, run_time=2)
        self.play(Write(result_texts))
        self.wait(2)

        # Clear addition results
        self.play(FadeOut(addition_title), FadeOut(result_texts))
        self.wait(1)

        # Reduction Operation - O(log n)
        reduction_title = Text("Reduction (Sum): O(log n) - Requires multiple steps", 
                              **font_style, color=RED)
        reduction_title.to_edge(DOWN)
        self.play(Write(reduction_title))
        self.wait(1)

        # Show initial values in blocks
        values = [sum(vector_a[i*block_size:(i+1)*block_size]) for i in range(num_blocks)]
        value_texts = VGroup()
        
        for i in range(num_blocks):
            text = Text(f"Sum: {values[i]}", **font_style, color=YELLOW)
            text.move_to(blocks[i].get_center()).shift(0.5*UP)
            value_texts.add(text)
        
        self.play(Transform(data_texts, value_texts))
        self.wait(1)

        # Step 1: First reduction (parallel within blocks)
        step1_text = Text("Step 1: Parallel reduction within each block", **font_style, color=RED)
        step1_text.to_edge(UP).shift(DOWN*1)
        # step1_text.next_to(reduction_title, UP)
        self.play(Write(step1_text))
        self.wait(1)

        # Show intermediate values
        intermediate_values = values.copy()
        intermediate_texts = VGroup()
        
        for i in range(num_blocks):
            text = Text(f"{values[i]}", **font_style, color=YELLOW)
            text.move_to(blocks[i].get_center())
            intermediate_texts.add(text)
        
        self.play(Transform(value_texts, intermediate_texts))
        self.wait(1)

        # Step 2: Reduce across blocks (logarithmic steps)
        step2_text = Text("Step 2: Reduce across blocks (requires sequential steps)", **font_style, color=RED)
        self.play(Transform(step1_text, step2_text))
        step2_text.to_edge(UP).shift(DOWN*1.8)
        self.wait(1)

        # Show reduction process
        arrows = VGroup()
        current_values = values
        
        while len(current_values) > 1:
            new_values = []
            new_texts = VGroup()
            new_arrows = VGroup()
            
            # Pair up values and show reduction
            for i in range(0, len(current_values), 2):
                if i+1 < len(current_values):
                    # Create arrow between pairs
                    arrow = Arrow(
                        blocks[i].get_right(), 
                        blocks[i+1].get_left(), 
                        color=RED, buff=0.1
                    )
                    new_arrows.add(arrow)
                    
                    # Calculate sum
                    pair_sum = current_values[i] + current_values[i+1]
                    new_values.append(pair_sum)
                    
                    # Create new text for the sum
                    text = Text(f"{pair_sum}", **font_style, color=YELLOW)
                    text.move_to((blocks[i].get_center() + blocks[i+1].get_center()) / 2 + DOWN*1)
                    new_texts.add(text)
            
            # If odd number of values, carry over the last one
            if len(current_values) % 2 == 1:
                new_values.append(current_values[-1])
                text = Text(f"{current_values[-1]}", **font_style, color=YELLOW)
                text.move_to(blocks[-1].get_center() + DOWN*1)
                new_texts.add(text)
            
            self.play(Create(new_arrows))
            self.wait(0.5)
            self.play(Transform(value_texts, new_texts), FadeOut(arrows))
            self.wait(1)
            
            current_values = new_values
            arrows = new_arrows

        # Final result
        final_result = Text(f"Final Sum: {current_values[0]}", **font_style, color=GREEN)
        final_result.move_to(DOWN*2.5)
        self.play(Write(final_result))
        self.wait(2)

        # Complexity comparison
        comparison_text = Text(
            "Vector Addition: O(1) - All elements processed in parallel\n"
            "Reduction: O(log n) - Requires multiple sequential steps",
            **font_style
        )
        comparison_text.to_edge(DOWN)
        
        self.play(
            FadeOut(step1_text),
            FadeOut(reduction_title),
            FadeOut(final_result),
            FadeOut(arrows),
            FadeOut(value_texts),
            FadeOut(blocks_title),
            FadeOut(data_texts),
        )
        
        self.play(blocks.animate.move_to(ORIGIN).scale(0.7))
        self.play(Write(comparison_text))
        self.wait(3)      # self.wait(0.5)


if __name__ == "__main__":
    import os
    # os.system(" manim -ql --resolution 1920,1080 parallel_cost.py VectorOpsComparison")
    os.system(" manim -qh --resolution 1920,1080 parallel_cost.py VectorOpsComparison")
