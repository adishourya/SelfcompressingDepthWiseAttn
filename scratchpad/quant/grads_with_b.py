import marimo

__generated_with = "0.11.13"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    import marimo
    return marimo, torch


@app.cell
def _(torch):
    def foo(half:float):
        x = torch.linspace(-3,3,50)
        x.requires_grad_()
        half = torch.tensor(half)
        half.requires_grad_()
        min = -half
        max = half - 1
        y = torch.clip(x, min, max)
        y.retain_grad()
        z = y.sum()
    
        z.backward()
        return half.grad

    return (foo,)


@app.cell
def _(marimo):
    half_slider = marimo.ui.slider(start=-10,stop = 10,step=0.1)
    half_slider
    return (half_slider,)


@app.cell
def _(foo, half_slider):
    half_grad = foo(half_slider.value)
    half_grad
    return (half_grad,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
