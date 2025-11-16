from manim import *

font_style = dict(font_size=18, font="Simple Nerd Font",color=BLACK)

class GenerateTable(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        # Table data
        table_data = [
            ["Model", "Size", "Top-1 (%) ↑", "Top-5 (%) ↑", "FLOPS (G) ↓", "Throughput (img/s) ↑"],
            ["EfficientViT", "0.98M", "71.3", "95.5", "7", "309"],
            ["DiffQ", "1.12M", "68.1", "92.3", "9", "281"],
            ["Ours", "0.88M", "73.4 (1st)", "98.0", "7", "338 (1st)"],
            ["EfficientViT", "2.41M", "76.2", "100", "24", "262"],
            ["DiffQ", "2.50M", "71.3", "100", "28", "276"],
            ["Ours", "1.98M", "75.8 (2nd)", "100", "24", "308 (1st)"],
            ["EfficientViT", "4.81M", "81.5", "100", "48", "183"],
            ["DiffQ", "4.92M", "78.1", "100", "54", "172"],
            ["Ours", "4.68M", "83.5 (1st)", "100", "51", "230 (1st)"]
        ]

        # Create table
        table = Table(
            table_data,
            include_outer_lines=True
        )

        # Set all lines to black with thinner stroke
        table.get_horizontal_lines().set_color(BLACK).set_stroke(width=1)
        table.get_vertical_lines().set_color(BLACK).set_stroke(width=1)

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
        
        # Scale the entire group
        everything.scale(0.40)

        # Center the group
        everything.move_to(ORIGIN)

        self.play(Create(everything))
        title = Text("Evaluation Results on CUB Dataset",**font_style)
        title.to_edge(DOWN)
        self.play(Write(title))
        self.wait(2)

if __name__ == "__main__":
    import os
    os.system("manim -qh --resolution 1920,1080 table_cub.py GenerateTable")
