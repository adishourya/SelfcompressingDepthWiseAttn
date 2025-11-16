
from manim import *

class GenerateTable(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        # Table data
        table_data = [
            ["Model", "Size", "mIoU ↑", "FLOPS (G) ↓", "Throughput (img/s) ↑"],
            ["EfficientViT", "4.68M", "44.1", "47", "255"],
            ["DiffQ", "4.92M", "40.1", "54", "241"],
            ["Ours", "4.28M", "43.7", "44", "282"]
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

        # Make "Ours" row red_e
        ours_row_index = 3  # Row index for "Ours" row (0-based, including header)
        if ours_row_index < len(table.get_rows()):
            row_cells = table.get_rows()[ours_row_index]
            for cell in row_cells:
                cell.set_color(RED_E)

        # Create a group containing everything
        everything = VGroup(table)
        
        # Scale the entire group to fit better
        everything.scale(0.6)

        # Center the group
        everything.move_to(ORIGIN)

        self.play(Create(everything))
        
        # Add title
        title = Text("Evaluation Results on ADE20k Dataset", font_size=24, color=BLACK)
        title.to_edge(DOWN)
        self.play(Write(title))
        self.wait(2)

if __name__ == "__main__":
    import os
    os.system("manim -qh --resolution 1920,1080 ade_table.py GenerateTable")
