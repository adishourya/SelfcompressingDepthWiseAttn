import marimo

__generated_with = "0.11.13"
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
def _(plt, torch):
    # clipping gradinet
    def foo():
        # x = torch.arange(-100.0,30.0)
        s = torch.tensor(-8.0)
        s.requires_grad_()
    
        x = torch.linspace(-2,2,100)
        x.requires_grad_()
    
        half = torch.tensor(0.501)
        half.requires_grad_()

        y = torch.clip(x,-1*half,half-1)
        y.retain_grad()
    
        z = torch.exp2(s) * y
        z.retain_grad()
        loss = z.sum()
        loss.backward()
        #-------------------------

        #----------------------BWD PASS---------
        #-------------
        # loss = z.sum()
        assert torch.all(z.grad == torch.ones_like(z))
    
        #-----------
        # y = exp2(s) *y
        # for y
        assert torch.all(y.grad == z.grad * torch.exp2(s))

        # for s
        our_sgrad = torch.sum(z.grad * z* torch.log(torch.tensor(2.0)))
        assert torch.allclose(s.grad , our_sgrad), f"{s.grad=}, {our_sgrad=}"

        #-----------
        # y = clip(x,-half,half-1)
        # for half
        ours_half_lj = torch.where(x<-1*half,-1,
                          torch.where(x>half-1,1,0))


        # but we can also have case where half is negative.
        # clipping does not work.. the gradient returns the size of the input.
        print("small half" if -1*half > half -1 else "ok")
        ours_half_lj =  (-1*half > half-1)*1 + (-1*half < half -1)*ours_half_lj
        ours_halfgrad = (y.grad * ours_half_lj).sum()
        assert torch.allclose(half.grad , ours_halfgrad)

        # for x
        ours_x_lj = torch.where(x< -1*half,0,
                               torch.where(x > half-1,0,1)
                               )
        ours_xgrad = y.grad * ours_x_lj
        assert torch.equal(x.grad , ours_xgrad)
    

        #----------
    
        #-----------------
        plt.hist(y.detach())
        plt.show()
    foo()
    return (foo,)


@app.cell
def _(steRound, torch):
    # clipping gradinet
    def bwd_pass(s,b):
        s = torch.tensor(s)
        b = torch.tensor(b)
        s.requires_grad_()
        b.requires_grad_()
    
        x = torch.linspace(-2,2,100)
        x.requires_grad_()
    
        x_scaled = x/ torch.exp2(s)
        x_scaled.retain_grad()


        half = torch.exp2(b-1)
        half.retain_grad()

        x_round = steRound(x_scaled)
        x_round.retain_grad()

        y = torch.clip(x_round,-1*half,half-1)
        y.retain_grad()
    
        z = torch.exp2(s) * y
        z.retain_grad()
        loss = z.sum()
        loss.backward()
        #-------------------------

    
        #---------------------------------------
        #----------------------BWD PASS---------
        #---------------------------------------
    
        #-------------
        # loss = z.sum()
        assert torch.all(z.grad == torch.ones_like(z))
    
        #-----------
        # z = exp2(s) *y
        # for y
        assert torch.all(y.grad == z.grad * torch.exp2(s))

        # for s (branch1)
        ours_sgrad1 = torch.sum(z.grad * z* torch.log(torch.tensor(2.0)))

        #-----------
        # y = clip(x_round,-half,half-1)
        # for half
        ours_half_lj = torch.where(x_round<-1*half,-1,
                          torch.where(x_round>half-1,1,0))


        # but we can also have case where half is negative.
        # clipping does not work.. the gradient returns the size of the input.
        print("small half" if -1*half > half -1 else "half ok")
        ours_half_lj =  (-1*half > half-1)*1 + (-1*half < half -1)*ours_half_lj
        ours_halfgrad = (y.grad * ours_half_lj).sum()
        assert torch.allclose(half.grad , ours_halfgrad, atol=1e-6) , f"{half.grad=} {ours_halfgrad=}"

        # for x_round
        ours_xround_lj = torch.where(x_round< -1*half,0,
                               torch.where(x_round > half-1,0,1)
                               )
        ours_xroundgrad = y.grad * ours_xround_lj
        assert torch.equal(x_round.grad , ours_xroundgrad)

        #---------
        # x_round = steround(x_scaled)
        assert torch.equal(x_scaled.grad, x_round.grad)
    

        #---------
        # half = exp2(b-1)
        ours_b_lj = half * torch.log(torch.tensor(2))
        ours_bgrad = half.grad * ours_b_lj

        #--------
        # x_scaled = x/ exp2(s)
        # for x
        our_xgrad = x_scaled.grad/ torch.exp2(s)
        assert torch.equal(our_xgrad, x.grad)

        # for s (branch2)
        ours_s_lj = x_scaled * -1* torch.log(torch.tensor(2.0))
        ours_sgrad2 = (x_scaled.grad * ours_s_lj).sum()

        ours_sgrad = ours_sgrad1+ ours_sgrad2
        assert torch.allclose(s.grad ,ours_sgrad, atol=1e-5), f"{ours_sgrad=}, {s.grad=} , {ours_sgrad - s.grad}"
    

        #----------
        print("ALL TESTS PASS")
    
    bwd_pass(s=-4.,b=1.)
    return (bwd_pass,)


@app.cell
def _(bwd_pass, torch):
    s_linspace = torch.linspace(2,-8,100)
    b_linspace = torch.linspace(-2,10,100)


    for exp_bit in s_linspace:
        for depth_bit in b_linspace:
            print(f"{exp_bit=},{depth_bit=}")
            bwd_pass(s=exp_bit, b = depth_bit)
    return b_linspace, depth_bit, exp_bit, s_linspace


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
