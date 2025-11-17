from manim import *

class GenerateTable(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        # Table data - all rows have same length
        table_data = [
            ["Model", "Size ↓", "Top-1 FP32 ↑", "Top-1 FP8 ↑", "Top-5 FP32 ↑", "Top-5 FP8 ↑", "FLOPs (G) ↓", "Throughput FP8 (img/s) ↑"],
            ["EfficientViT", "0.98M", "71.3", "66.1", "95.5", "90.2", "7", "309"],
            ["DiffQ", "1.12M", "68.1", "68.0", "92.3", "92.5", "9", "281"],
            ["Ours", "0.88M", "73.4 (1st)", "73.4", "98.0", "97.2", "7", "338 (1st)"],
            ["EfficientViT", "2.41M", "76.2", "72.0", "100", "96.5", "24", "262"],
            ["DiffQ", "2.50M", "71.3", "70.0", "100", "94.1", "28", "276"],
            ["Ours", "1.98M", "75.8 (2nd)", "74.9", "100", "100", "24", "308 (1st)"],
            ["EfficientViT", "4.81M", "81.5", "75.5", "100", "98.0", "48", "183"],
            ["DiffQ", "4.92M", "78.1", "76.4", "100", "96.2", "54", "172"],
            ["Ours", "4.68M", "83.5 (1st)", "83.4", "100", "99.3", "51", "230 (1st)"]
        ]

        # Create table
        table = Table(
            table_data,
            include_outer_lines=True
        )

        # Set all lines to black with thinner stroke
        table.get_horizontal_lines().set_color(BLACK).set_stroke(width=0.7)
        table.get_vertical_lines().set_color(BLACK).set_stroke(width=0.7)

        # Set default text color to black
        entries = table.get_entries()
        for entry in entries:
            entry.set_color(BLACK)

        # Make header text blue_e
        header_cells = table.get_rows()[0]  # First row is header
        for cell in header_cells:
            cell.set_color(BLUE_E)

        # Make "Ours" rows red_e
        ours_row_indices = [3, 6, 9]  # Row indices for "Ours" rows (0-based, including header)
        for row_idx in ours_row_indices:
            if row_idx < len(table.get_rows()):
                row_cells = table.get_rows()[row_idx]
                for cell in row_cells:
                    cell.set_color(RED_E)

        # Create a group containing everything
        everything = VGroup(table)
        
        # Scale the entire group to fit better
        everything.scale(0.32)

        # Center the group
        everything.move_to(ORIGIN)

        self.play(Create(everything))
        
        # Add title
        title = Text("Evaluation Results on CUB 2011 Dataset", font_size=20, font="Simple Nerd Font", color=BLACK)
        title.to_edge(DOWN)
        self.play(Write(title))
        self.wait(2)

if __name__ == "__main__":
    import os
    os.system("manim -qh --resolution 1920,1080 table_cub.py GenerateTable")
