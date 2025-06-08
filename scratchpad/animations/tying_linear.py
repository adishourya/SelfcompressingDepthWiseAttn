import manim
from manim.utils.debug import VGroup
import numpy as np
np.random.seed(1)
import os

class Matrix(manim.Scene):
    def construct(self):
        # Title
        title = manim.Text("Weight Tying for Feature Selection", font="Simple Nerd Font").scale(0.2)
        title.to_edge(manim.UL)
        self.add(title)

        # Subtitle / idea text
        idea = manim.Text(
            "Prunable Features through nn.Linear",
            font="Simple Nerd Font",
            font_size=22,
            color=manim.YELLOW,
        )
        idea.to_edge(manim.UP, buff=1)
        self.add(idea)

        # Prunable feature shapes
        p_features = [manim.Circle(radius=0.2), manim.Square(0.4), manim.Star().scale(0.2)]

        # Input matrix Xi with integers and shapes
        xi = [
            [manim.Integer(1), manim.Integer(2), manim.Integer(3), p_features[0], manim.Integer(4)],
            [manim.Integer(5), manim.Integer(6), manim.Integer(7), p_features[1], manim.Integer(8)],
            [manim.Integer(9), manim.Integer(1), manim.Integer(2), p_features[2], manim.Integer(3)],
        ]
        Xi = manim.MobjectMatrix(xi)
        xi_label = manim.MathTex(r"X_i").next_to(Xi, manim.DOWN)

        # Weight matrix wi - 5x6 matrix of w_ij
        w_expr = [[manim.MathTex(f"w_{{{i}{j}}}") for j in range(6)] for i in range(5)]
        wi = manim.MobjectMatrix(w_expr)
        wi_label = manim.MathTex(r"W_i")
        wi_label.next_to(wi,manim.DOWN*2)

        wi_1g = manim.VGroup(wi, wi_label).scale(0.8)

        # Group prunable weights (4th row: i=3)
        pr_weights = w_expr[3]
        prw_g = manim.VGroup(*pr_weights)

        # Modified weights where 5th row (i=4) is q5(w_5j)
        w_expr2 = [
            [
                manim.MathTex(f"q_4(w_{{5{j}}})", color=manim.YELLOW, font_size = 25) if i == 3 else manim.MathTex(f"w_{{{i}{j}}}")
                for j in range(6)
            ]
            for i in range(5)  # 5 rows total (0 to 4)
        ]
        wi2 = manim.MobjectMatrix(w_expr2)
        wi_2g = manim.VGroup(wi2, wi_label.copy()).scale(0.8)

        # Define distinct colors for each row
        row_colors = [manim.RED, manim.BLUE, manim.GREEN, manim.YELLOW, manim.PURPLE]

        # Modified weights where 5th row (i=4) is zeroed out
        w_expr3 = [
            [
                manim.MathTex("0", color=row_colors[i]) if i == 3 else manim.MathTex(f"q_{{{i}{j}}}",color=row_colors[i])
                for j in range(6)
            ]
            for i in range(5)
        ]
        wi3 = manim.MobjectMatrix(w_expr3)
        wi_3g = manim.VGroup(wi3, wi_label.copy()).scale(0.8)


        # Modified weight matrix: q_i(w_{ij}) with different row colors
        w_expr4 = [
            [
                manim.MathTex(f"q_{{{i}}}(w_{{{i}{j}}})", color=row_colors[i], font_size=25)
                for j in range(6)
            ]
            for i in range(5)
        ]

        # Create the matrix and label group
        wi4 = manim.MobjectMatrix(w_expr4)
        wi4_label = manim.MathTex(r"W_i").next_to(wi4, manim.DOWN)
        wi_4g = manim.VGroup(wi4, wi4_label).scale(0.8)

        # Group input matrix and label
        Xg = manim.VGroup(Xi, xi_label)

        # Add input matrix and label with fade in
        self.play(manim.FadeIn(Xi), manim.FadeIn(xi_label))
        title1 = manim.Text("Intermediate representation at some time step i\n Examples are laid out as independent rows",color=manim.GREEN,font_size=18,font="Simple Nerd Font")
        title1.to_edge(manim.DOWN)
        self.add(title1)
        self.wait(2)

        # Indicate prunable features
        p_grpup = manim.VGroup(*p_features)
        title2 = manim.Text("Noisy Feature in 4th Channel",color=manim.RED,font_size=18,font="Simple Nerd Font")
        self.wait(1)
        title2.move_to(title1.get_center())
        self.play(manim.Transform(title1,title2))
        self.play(manim.Indicate(p_grpup, color=manim.RED))
        self.wait(2)

        # Shift input matrix group left and scale down
        self.play(Xg.animate.shift(manim.LEFT * 4))
        self.play(manim.ScaleInPlace(Xg, 0.8))

        # Position wi_1g to the right of Xg and add
        wi_1g.next_to(Xg, manim.RIGHT)
        self.add(wi_1g)

        # Shift wi_1g further right
        self.play(wi_1g.animate.shift(manim.RIGHT * 0.5))

        title3 = manim.Text("Then the associated dual basis can be compressed\nHere the 4th row",color=manim.YELLOW,font_size=18,font="Simple Nerd Font")
        self.wait(1)
        title3.move_to(title1.get_center())
        self.play(manim.Transform(title1,title3))
        self.wait(1)


        # Indicate prunable weights
        self.play(manim.Indicate(prw_g), manim.Indicate(p_grpup))
        self.wait(1)

        title4 = manim.Text("Broadcasting quant function along the rows",color=manim.BLUE,font_size=18,font="Simple Nerd Font")
        self.wait(1)
        title4.move_to(title1.get_center())
        self.play(manim.Transform(title1,title4))


        # IMPORTANT: move wi_2g and wi_3g to wi_1g's current position BEFORE transforming
        wi_2g.move_to(wi_1g.get_center())
        wi_3g.move_to(wi_1g.get_center())
        wi_4g.move_to(wi_1g.get_center())


        # Transform wi_1g to wi_2g in place
        self.play(manim.Transform(wi_1g, wi_4g))
        self.wait(1)

        # Transform wi_1g to wi_3g in place
        self.play(manim.Transform(wi_1g, wi_3g))
        title5 = manim.Text("Exp and depth bits can adjust during training\ndepending on the influence of scale and precision needed for independent features",color=manim.GREEN,font_size=18,font="Simple Nerd Font")
        self.wait(1)
        title5.move_to(title1.get_center())
        self.play(manim.Transform(title1,title5))
        self.wait(4)


if __name__ == "__main__":
    os.system("manim -qh --resolution 1920,1080 tying_linear.py Matrix")
