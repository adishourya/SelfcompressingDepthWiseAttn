import os
from turtle import left
from typing_extensions import runtime
from manim import *

# config.background_color = WHITE
# config["background_color"] = WHITE
import numpy as np

def qbits(x, b=6.0, e=-8.0):
    b = b if b > 0 else 0
    x_scaled = x / np.exp2(e)
    x_clipped = np.clip(x_scaled, -1 * np.exp2(b - 1), np.exp2(b - 1) - 1)
    x_round = np.round(x_clipped)
    result = np.exp2(e) * x_round
    return (result, x_scaled, x_clipped, x_round),(-1*np.exp2(b-1), np.exp2(b-1)-1)

# class QuantizationAnimation(MovingCameraScene):
class QuantizationAnimation(MovingCameraScene):
    
    def construct(self):
        # dont change this... [took a while to find a good value for animation]
        e_bit = -3.5
        b_bit = 3.5
        x_input = np.linspace(-2,2,100)
        y_init = x_input
        equation = MathTex(
        r"q(x) = 2^{e} round(clip(\frac{x}{2^{e}}, -2^{b-1}, 2^{b-1} -1))",
        color=WHITE
        )

        (y_result,y1,y2,y3),(cl_l,cl_r) = qbits(x_input,b_bit,e_bit)
        print("???",cl_l,cl_r)

        ax1 = Axes(
            x_range=[x_input[0], x_input[-1], 5],
            y_range=[y_init[0], y_init[-1], 5],
            x_length=5,
            y_length=5,
            x_axis_config={"numbers_to_include": [x_input[0],x_input[-1]]},
            y_axis_config={"numbers_to_include":[y_init[0],y_init[-1]]},
            tips=True,
            )
        ax1_copy =ax1.copy()

        ax_labels = ax1.get_axis_labels(x_label="x", y_label="q(x)")

        # code
        code = """
            def qbits(x, b, e):
                b = torch.relu(b)
                x_scaled = x / torch.exp2(e)
                x_clipped = torch.clip(x_scaled, -1 * torch.exp2(b - 1), torch.exp2(b - 1) - 1)
                x_round = np.round(x_clipped)
                result = np.exp2(e) * x_round
                return result
        """

        rendered_code = Code(
        code_string=code,
        language="python",
        ).scale(0.3)
        rendered_code.to_corner(DR)

        

        # first step
        # plane1=NumberPlane(x_range=(-10,10,1),
        #                    y_range=(-50,50,1),
        #                    background_line_style={"stroke_width": 1, "stroke_color": GRAY},)

        plane1=NumberPlane(x_range=(x_input[0],x_input[-1]),
                           y_range=(y_init[0],y_init[-1]),
                           background_line_style={"stroke_width": 1, "stroke_color": GRAY},)
        graph1 = ax1.plot_line_graph(x_input, y_init, line_color=ORANGE,vertex_dot_radius=0)

        # scale it down
        ax2 = Axes(
            x_range=[x_input[0],x_input[-1]],
            y_range=[y1[0], y1[-1]],
            x_length=5,
            y_length=7,
            x_axis_config={"numbers_to_include": x_input[::len(x_input//10)]},
            y_axis_config={"numbers_to_include": y1[::len(y1//10)]},
            tips=True,
        )
        graph2 = ax2.plot_line_graph(x_input,y1,line_color=RED,vertex_dot_radius=0)

        #clip it 
        ax3 = Axes(
            x_range=[x_input[0],x_input[-1]],
            y_range=[y1[0], y1[-1]],
            x_length=5,
            y_length=7,
            x_axis_config={"numbers_to_include": x_input[::len(x_input//10)]},
            y_axis_config={"numbers_to_include": y1[::len(y1//10)]},
            tips=True,
        )
        graph3 = ax3.plot_line_graph(x_input,y2,line_color=YELLOW,vertex_dot_radius=0)

        #quantize
        ax4 = Axes(
            x_range=[x_input[0],x_input[-1]],
            y_range=[y1[0], y1[-1]],
            x_length=5,
            y_length=7,
            x_axis_config={"numbers_to_include": x_input[::len(x_input//10)]},
            y_axis_config={"numbers_to_include": y1[::len(y1//10)]},
            tips=True,
        )
        graph4 = ax4.plot_line_graph(x_input,y3,line_color=GREEN,vertex_dot_radius=0)

        #rescale it back
        ax5 = Axes(
            x_range=[x_input[0],x_input[-1]],
            y_range=[y_result[0], y_result[-1]],
            x_length=5,
            y_length=5,
            x_axis_config={"numbers_to_include": x_input[::len(x_input//10)]},
            y_axis_config={"numbers_to_include": y1[::len(y1//10)]},
            tips=True,
        )
        graph5 = ax5.plot_line_graph(x_input,y_result,line_color=PURPLE,vertex_dot_radius=0)



        # linspaced inputs
        title = Text(f"Quantization Function\n{b_bit=}, {e_bit=}",font_size=14,font="Simple Nerd Font", color = GRAY_A)
        title.to_corner(UR)

        author = Text("-Adi",font_size=5,font="Simple Nerd Font")
        author.to_corner(DL)


        # graph1_title =Text("Linspaced Input (-3,3)",font_size=20,font="Simple Nerd Font",color=ORANGE).to_corner(UL)
        # graph2_title =Text("Scale UP \n *1/torch.exp2(e) ",font_size=20,font="Simple Nerd Font",color=RED).to_corner(UL)
        # graph3_title =Text("CLIP \n (-1*torch.exp2(b-1), torch.exp2(b-1)-1)",font="Simple Nerd Font",font_size=20,color=YELLOW).to_corner(UL)
        # graph4_title =Text("Quantize torch.round (STE)",font_size=20,font="Simple Nerd Font",color=GREEN).to_corner(UL)
        # graph5_title =Text("Scale Down * torch.exp2(e)",font_size=20,font="Simple Nerd Font",color=PURPLE).to_corner(UL)
        
        graph1_title = Text("Linspaced Input (-3,3)", font_size=20, font="Simple Nerd Font", color=ORANGE).next_to(graph1,LEFT)
        graph2_title = MathTex(r"\text{Scale} \frac{1}{2^e}", font_size=20,  color=RED).next_to(graph2,LEFT)
        graph3_title = MathTex(r"\text{clip}(-2^{b-1}, 2^{b-1} - 1)", font_size=20,  color=YELLOW).next_to(graph3,LEFT)
        # Alternatively, if you want to keep the torch code look:
        # graph3_title = Text("CLIP \n (-1*torch.exp2(b-1), torch.exp2(b-1)-1)", font="Simple Nerd Font", font_size=20, color=YELLOW).next_to(graph3)
        graph4_title = Text("Quantize torch.round\n(STE)", font_size=20,font="Simple Nerd Font",  color=GREEN).next_to(graph4,LEFT)
        graph5_title = MathTex(r"\text{Scale Down}  2^e", font_size=20,  color=PURPLE).next_to(graph5,LEFT)


        # self.add(equation)
        self.play(Write(equation))
        self.play(ScaleInPlace(equation,0.5))
        self.play(equation.animate.to_corner(UL))

        self.play(Write(plane1),Create(ax1),Write(ax_labels),Write(title),Write(author))
        self.wait(2)
        self.play(Write(graph1),Write(graph1_title))
        self.wait(2)
        # scaled up with exp bits
        self.play(Transform(ax1,ax2),ScaleInPlace(plane1,np.exp2(e_bit)),Transform(graph1,graph2),Transform(graph1_title,graph2_title))
        self.wait(2)
        # clipped



        clip_line1 = ax2.plot(lambda x:cl_l,color=YELLOW)
        clip_line2 = ax2.plot(lambda x:cl_r,color=YELLOW)
        self.play(Transform(graph1_title,graph3_title))
        self.play(Create(clip_line1),Create(clip_line2),runtime=0.5)
        self.play(Uncreate(clip_line1),Uncreate(clip_line2), runtime=0.5)

        self.play(Transform(graph1,graph3))

        self.wait(2)
        # quantize
        self.play(Transform(graph1_title,graph4_title))
        # + zoom in to show quantization near origin
        self.play(self.camera.frame.animate.scale(0.1).move_to(ORIGIN), run_time=1)
        self.play(Transform(graph1,graph4))
        self.wait(3)
        self.play(self.camera.frame.animate.scale(10).move_to(ORIGIN), run_time=1)
        self.wait(2)
        # rescale
        self.play(Transform(ax1,ax1_copy),ScaleInPlace(plane1,1/np.exp2(e_bit)),Transform(graph1,graph5),Transform(graph1_title,graph5_title))
        self.wait(5)


if __name__ == "__main__":
    # os.system("manim -qh --resolution 1920,1080 quant_animation.py QuantizationAnimation")
    os.system("manim -qh --resolution 960,1080 quant_animation.py QuantizationAnimation")
    # os.system("manim -qm --resolution 960,1080 quant_animation.py QuantizationAnimation")
