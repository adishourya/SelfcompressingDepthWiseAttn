from manim import *

class GenerateTable(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        # Table data - all rows have same length
        table_data = [
            ["Model", "Size ↓", "Top-1 FP32 ↑", "Top-1 FP8 ↑", "Top-5 FP32 ↑", "Top-5 FP8 ↑", "FLOPs (G) ↓", "Throughput FP8 (img/s) ↑"],
            ["EfficientViT", "2.41M", "44.4", "41.2", "65.6", "52.1", "23", "291"],
            ["DiffQ", "2.50M", "36.0", "35.8", "41.1", "40.4", "28", "270"],
            ["Ours", "2.13M", "43.5 (2nd)", "43.5", "65.3", "63.1", "25", "324 (1st)"],
            ["EfficientViT", "4.81M", "52.1", "44.9", "80.2", "69.5", "48", "228"],
            ["DiffQ", "4.92M", "38.4", "38.3", "65.2", "65.5", "54", "213"],
            ["Ours", "4.68M", "55.7 (1st)", "54.9", "81.0", "81.3", "51", "294 (1st)"]
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
        ours_row_indices = [3, 6]  # Row indices for "Ours" rows (0-based, including header)
        for row_idx in ours_row_indices:
            if row_idx < len(table.get_rows()):
                row_cells = table.get_rows()[row_idx]
                for cell in row_cells:
                    cell.set_color(RED_E)

        # Create a group containing everything
        everything = VGroup(table)
        
        # Scale the entire group to fit better
        everything.scale(0.35)

        # Center the group
        everything.move_to(ORIGIN)

        self.play(Create(everything))
        
        # Add title
        title = Text("Evaluation Results on Country211 Dataset", font_size=20,font="Simple Nerd Font", color=BLACK)
        title.to_edge(DOWN)
        self.play(Write(title))
        self.wait(2)

if __name__ == "__main__":
    import os
    os.system("manim -qh --resolution 1920,1080 country_table.py GenerateTable")
