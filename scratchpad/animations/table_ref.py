
from manim import *

class GenerateTable(Scene):
    def construct(self):
        self.camera.background_color= WHITE

        table = Table(
            [["First", "Second"],
            ["Third","Fourth"]],
            row_labels=[Text("R1"), Text("R2")],
            col_labels=[Text("C1"), Text("C2")])

        
        table.get_horizontal_lines().set_color(BLACK)
        table.get_vertical_lines().set_color(BLACK)

        
        ent = table.get_entries_without_labels()
        colors = [BLUE, GREEN, YELLOW, RED]
        for k in range(len(colors)):
            ent[k].set_color(colors[k])

        highlight = table.get_highlighted_cell((2,2), color=GREEN)
        table.add_to_back(highlight)

        lab = table.get_labels()
        colors = [BLUE, GREEN, YELLOW, RED]

        for k in range(len(colors)):
            lab[k].set_color(colors[k])
                # table.get_entries_without_labels((2,2)).rotate(PI)
        # self.add(table)
        table.add(SurroundingRectangle(table.get_rows()[1]))
        self.play(Write(table))
        # self.play(SurroundingRectangle(table.get_rows()[1]))

if __name__ == "__main__":
    import os
    os.system("manim -qh --resolution 1920,1080 table_cub.py GenerateTable")
