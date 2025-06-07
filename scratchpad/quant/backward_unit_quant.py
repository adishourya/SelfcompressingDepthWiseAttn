import marimo

__generated_with = "0.11.21"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    import matplotlib.pyplot as plt

    class STERound(torch.autograd.Function):
        @staticmethod
        def forward(ctx,x):
            return torch.round(x)

        @staticmethod
        def backward(ctx,upstream):
            return upstream

    steRound = STERound.apply

    return STERound, plt, steRound, torch


@app.cell
def _(steRound, torch):
    def inspect_gradient():
        x = torch.linspace(-3,3,20)
        print(f"{x=}")
        x.requires_grad_()

        b = torch.tensor(3.0)
        b.requires_grad_()
        brange = torch.exp2(b-1)
        brange.retain_grad()

        e = torch.tensor(-5.0)
        e.requires_grad_()

        x_scaled = x/torch.exp2(e)
        x_scaled.retain_grad()
        print(f"{x_scaled=}")

        x_clipped = torch.clip(x_scaled,-1*brange, brange-1)
        x_clipped.retain_grad()
        print(f"clipped range {-1*torch.exp2(b-1)}, {torch.exp2(b-1)-1}")
        print(f"{x_clipped=}")


        x_round = steRound(x_clipped)
        x_round.retain_grad()
        print(f"{x_round=}")

        result = torch.exp2(e) * x_round
        result.retain_grad()
        print(f"{result=}")

        loss = result.sum()
        loss.backward()

        # unit tests
        print("==========unit_tests===============")
        our_resultGrad = torch.ones_like(result)
        print(f"{result.grad=}\n {our_resultGrad=}")
        assert torch.allclose(result.grad, our_resultGrad), "hein?"
        print(f"Test passed.")

        our_roundGrad = torch.ones_like(result)*torch.exp2(e)
        print(f"{x_round.grad=}\n{our_roundGrad=}")
        assert torch.allclose(x_round.grad, our_roundGrad), "hein?"
        print("Test Passed")

        de_branch1 = result.grad * result * torch.log(torch.tensor(2.0))

        print(f"{x_clipped.grad=}\n{x_round.grad=}")
        assert torch.equal(x_clipped.grad,x_round.grad),"hein?"
        print("Test Passed")

        low_range = -1*torch.exp2(b-1)
        high_range = torch.exp2(b-1)-1
        dx_scaled_local = torch.where(((x_scaled >= low_range)&(x_scaled<=high_range)),1.0,0.0)
        our_xscaledGrad = x_clipped.grad * dx_scaled_local
        print(f"{x_scaled.grad=}\n{our_xscaledGrad=}")
        assert torch.allclose(x_scaled.grad, our_xscaledGrad), "hein?"
        print("Test Passed")

        our_xGrad = x_scaled.grad * (1/torch.exp2(e))
        print(f"{x.grad=}\n{our_xGrad=}")
        assert torch.allclose(x.grad,our_xGrad),"hein?"
        print("Test Passed")

        brangegrad_branch1= torch.sum(torch.where((x_scaled<=low_range),-1.0,0.0)) 
        brangegrad_branch2= torch.sum(torch.where((x_scaled>=high_range),1.0,0.0)) 
        local_brangeGrad = (brangegrad_branch1 + brangegrad_branch2)
        our_brangeGrad = torch.sum(x_scaled.grad * local_brangeGrad)
        print(f"{brange.grad=}{our_brangeGrad=}")
        assert torch.allclose(brange.grad,our_brangeGrad),"Hein?"
        print("Test Passed")

        our_bGrad = brange.grad* brange * torch.log(torch.tensor(2))
        print(f"{our_bGrad=} {b.grad=}")
        assert torch.allclose(b.grad, our_bGrad) , "Hein?"
        print("Test Passed")

        de_branch2 = x_scaled.grad *x_scaled* -1*torch.log(torch.tensor(2.0))
        print(f"{de_branch1=}\n{de_branch2=}")
        de = de_branch1 + de_branch2
        print(f"{e.grad=}\n{de.sum()=}")
        assert torch.allclose(e.grad,de.sum()),"Hein"
        print("Test Passed")
        print("All Test Passed!")

    inspect_gradient()
    return (inspect_gradient,)


@app.cell
def _(steRound, torch):
    def inspect_gradient_grid(b_val: float, e_val: float):
        x = torch.linspace(-3, 3, 20)
        x.requires_grad_()

        b = torch.tensor(b_val)
        b.requires_grad_()
        brange = torch.exp2(b - 1)
        brange.retain_grad()

        e = torch.tensor(e_val)
        e.requires_grad_()

        x_scaled = x / torch.exp2(e)
        x_scaled.retain_grad()

        x_clipped = torch.clip(x_scaled, -1 * brange, brange - 1)
        x_clipped.retain_grad()

        # use the actual steRound defined earlier
        x_round = steRound(x_clipped)
        x_round.retain_grad()

        result = torch.exp2(e) * x_round
        result.retain_grad()

        loss = result.sum()
        loss.backward()

        # unit tests
        our_resultGrad = torch.ones_like(result)
        assert torch.allclose(result.grad, our_resultGrad), "result.grad check failed"

        our_roundGrad = torch.ones_like(result) * torch.exp2(e)
        assert torch.allclose(x_round.grad, our_roundGrad), "x_round.grad check failed"

        assert torch.equal(x_clipped.grad, x_round.grad), "x_clipped.grad != x_round.grad"

        low_range = -1 * torch.exp2(b - 1)
        high_range = torch.exp2(b - 1) - 1
        dx_scaled_local = torch.where(((x_scaled >= low_range) & (x_scaled <= high_range)), 1.0, 0.0)
        our_xscaledGrad = x_clipped.grad * dx_scaled_local
        assert torch.allclose(x_scaled.grad, our_xscaledGrad), "x_scaled.grad check failed"

        our_xGrad = x_scaled.grad * (1 / torch.exp2(e))
        assert torch.allclose(x.grad, our_xGrad), "x.grad check failed"

        brangegrad_branch1 = torch.sum(torch.where((x_scaled <= low_range), -1.0, 0.0))
        brangegrad_branch2 = torch.sum(torch.where((x_scaled >= high_range), 1.0, 0.0))
        local_brangeGrad = (brangegrad_branch1 + brangegrad_branch2)
        our_brangeGrad = torch.sum(x_scaled.grad * local_brangeGrad)
        assert torch.allclose(brange.grad, our_brangeGrad), f"brange.grad check failed, difference : {brange.grad - our_brangeGrad}"

        our_bGrad = brange.grad * brange * torch.log(torch.tensor(2.0))
        assert torch.allclose(b.grad, our_bGrad), "b.grad check failed"

        de_branch1 = result.grad * result * torch.log(torch.tensor(2.0))
        de_branch2 = x_scaled.grad * x_scaled * -1 * torch.log(torch.tensor(2.0))
        de = de_branch1 + de_branch2
        assert torch.allclose(e.grad, de.sum()), f"e.grad check failed, difference : {e.grad - de.sum()}"

        print(f"✅ All tests passed for b={b_val:.2f}, e={e_val:.2f}")

    return (inspect_gradient_grid,)


@app.cell
def _(inspect_gradient_grid, torch):
    import numpy as np

    b_vals = torch.linspace(0.1, 8.0, 100)
    e_vals = torch.linspace(-10.0, -2.0, 100)

    fail_map = torch.zeros((len(b_vals), len(e_vals)))

    for i, b in enumerate(b_vals):
        for j, e in enumerate(e_vals):
            try:
                inspect_gradient_grid(float(b), float(e))  # your test function
                fail_map[i, j] = 1  # pass
            except Exception:
                fail_map[i, j] = 0  # fail

    return b, b_vals, e, e_vals, fail_map, i, j, np


@app.cell
def _(b_vals, e_vals, fail_map, plt, torch):
    B, E = torch.meshgrid(b_vals, e_vals, indexing="ij")

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(B.numpy(), E.numpy(), fail_map.numpy(), levels=1, cmap="Blues", alpha=0.8)
    plt.colorbar(label="Test Failed (0 = fail, 1 = pass)")
    plt.xlabel("b")
    plt.ylabel("e")
    plt.title("Contour Plot of Test Failures in (b, e) Grid")
    plt.show()

    return B, E, contour


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
