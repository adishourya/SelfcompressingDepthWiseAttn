# keep comet ml import above torch [comet's documentation]
import os
os.getenv("comet_api")

import datetime
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs("assets", exist_ok=True)

from comet_ml import start
from comet_ml.integration.pytorch import log_model,watch


import torch
from qmodules.QConv import Qconv
from qmodules.QLinear import QlinearMLP
from tqdm import tqdm

class QTrainer:
    """
    Quantized Trainer class
    """
    def __init__(self,
                 model,
                 train_loader,
                 eval_loader,
                 dtype=torch.float32,
                 amp_dtype =torch.float32,
                 compression_gamma = 0.1,
                 pbar_track_freq = 50,
                 eval_track_freq = 5,
                 logging=False,
                 comet_username="adishourya",
                 tag="low_bval",
                 project_name="convolution_compressing"):

        self.model = model()
        self.train_loader =train_loader
        self.eval_loader = eval_loader
        self.pbar_track_freq = pbar_track_freq
        self.eval_track_freq = eval_track_freq


        assert isinstance(dtype,torch.dtype),"Hein? unrecognized dtype"
        assert isinstance(amp_dtype,torch.dtype),"Hein? unrecognized dtype"
        self.dtype = dtype
        # we will keep this for H100...
        self.amp_dtype = amp_dtype
        self.eps = torch.finfo(self.dtype).eps
        self.logging = logging

        if self.logging:
            # print("logging costs around 0.5 secs more per iteration.")
            self.comet_username = comet_username
            self.tag = tag
            self.project_name = project_name
            self._setup_logging()
        

        # does not work on my machine with less precision...
        if self.dtype == torch.float32:
            # do we actually need high precision matmul ??
            torch.set_float32_matmul_precision("high")

        print(f"Training with {self.dtype=}")

        self.scaler = torch.amp.GradScaler()
        # no need to do to dtype
        self.model = self.model.to(self.dtype)
        self.model = torch.compile(self.model)
        # dont try it on cpu!
        self.model.to("cuda")

        self.track_decay = []
        self.track_activekernels = []
        self.track_activedualbasis = []
        self.track_loss = []

    
        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            weight_decay=1e-3)
    
        self.gamma = compression_gamma # should be around 0.05 or something.. compression factor
        # we need to calculate total number of parameters at initialization (papar calls it N)
        # here since everything is trainable

        self.tot_init = sum(p.numel() for group in self.optim.param_groups for p in group['params'] if p.requires_grad)
        self.qtot_init = self._qlayersize() 
        self.tot_qparams = torch.sum(torch.tensor([p_weight.numel() for p,p_weight in self.model.named_parameters() if "_bit" in p]))
        self.bits_summary = self.model_bit_depth_summary() 
    
        print(f"Total Parameters {self.tot_init=}")
        print(f"Layer Size at Init: {self.qtot_init}")
        print(f"of which compression are :{self.tot_qparams=}")
        print(f"compression factor at init {self.gamma * self._qlayersize()}")
        print(f"Model ULP: {self.bits_summary=}")

        if self.logging:
            self.experiment.log_metric(name="Init Parameters", value=self.tot_init)
            self.experiment.log_metric(name="Total Qparams ", value=self.tot_qparams)
            self.experiment.log_metric(name="Total Layersize ", value=self.qtot_init)


    def ulp(self,x: torch.Tensor):
        """
        Units in Last place measurement.
        From : https://en.wikipedia.org/wiki/Unit_in_the_last_place
        is the distance between the two closest straddling floating-point
        >>> x = 1.0
        >>> p = 0
        >>> while x != x + 1:
        ...   x = x * 2
        ...   p = p + 1
        ... 
        >>> x
        9007199254740992.0
        >>> p
        53
        >>> x + 2 + 1
        9007199254740996.0

        """
        x = torch.abs(x)

        # Integer part bit depth... this is easy
        int_bits = torch.ceil(torch.log2(torch.clip(x, min=1.0)))
        int_bits = torch.clip(int_bits,min=0)
        # ULP scale (relative rounding resolution)
        ulp_scale = torch.clip(x * self.eps, min=self.eps)

        # Fractional error: how far off is x from nearest value (simulate quant error)
        residual = torch.abs(x - x.round())
        residual = torch.clip(residual, min=self.eps)
        ratio = torch.clip(residual/ulp_scale,max=1.0,min=self.eps)
        frac_bits = torch.ceil(-torch.log2(ratio))

        total_bits = (int_bits + frac_bits + 1).sum()
        return total_bits

    def model_bit_depth_summary(self):
        """
        Per weight not individual tensor
        """
        total_bits = 0
        for _, param in self.model.named_parameters():
            total_bits += self.ulp(param)
        return total_bits

    def _setup_logging(self):
        self.experiment = start(
          api_key=os.getenv("comet_api"),
          project_name=self.project_name,
          workspace=self.comet_username
        )
        self.experiment.add_tag(self.tag)
        # watch weights [to be precise i want to watch sparsity.. but we will see that later]
        # this is costly
        watch(self.model)


    def _qlayersize(self):
        size_conv = torch.sum(torch.tensor([layer.size_layer() for layer in self.model.modules() if isinstance(layer,Qconv)]))
        size_lin =  torch.sum(torch.tensor([layer.size_layer() for layer in self.model.modules() if isinstance(layer,QlinearMLP)]))
        # [print("->",layer,layer.size_layer()) for layer in self.model.modules() if isinstance(layer,QlinearMLP)]
        # print(size_lin,size_conv)
        return (size_conv + size_lin)


    def _activekernelscount(self):
        kernel_counts = dict()
        for name,layer in self.model.named_modules():
            if isinstance(layer,Qconv):
                depths = torch.relu(layer.depth_bit)
                # count nnz depth bits
                count =torch.sum(torch.where(depths>0,1,0)).item()
                kernel_counts[name] = count
        return kernel_counts

    def _activeLinearDualbasis(self):
        row_counts = dict()
        for name,layer in self.model.named_modules():
            if isinstance(layer,QlinearMLP):
                depths = torch.relu(layer.depth_bit)
                # count nnz depth bits
                count = torch.sum(torch.where(depths>0,1,0)).item()
                row_counts[name] = count
        return row_counts

    def _save_checkpoint(self):
        print("Saving")
        model_checkpoint = {
                ""
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optim.state_dict(),
                'loss':self.track_loss[-1]
                }
        return model_checkpoint


    def _track(self,loss,activekernels,activeduals,bit_decay,epoch):
        """
        append loss and active kernel stats.
        optionally loggs them to comet_ml
        """
        self.track_loss.append(loss)
        self.track_activekernels.append(activekernels.values())
        self.track_activedualbasis.append(activeduals.values())
        self.track_decay.append(bit_decay)
        if self.logging:
             self.experiment.log_metric(name="loss", value=loss, step=epoch)
             self.experiment.log_metric(name="decay", value=bit_decay, step=epoch)
             self.experiment.log_metric(name="activekernels", value=sum(activekernels.values()), step=epoch)
             self.experiment.log_metric(name="activeDualBasis", value=sum(activeduals.values()), step=epoch)

    @torch.no_grad
    def eval_run(self):
        # double sure !!
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        for batch_img, batch_label in self.eval_loader:
            batch_img = batch_img.to("cuda").to(self.dtype)
            batch_label = batch_label.to("cuda")
            out = self.model(batch_img)
            bit_decay = self._qlayersize()/self.qtot_init
            total_loss += torch.nn.functional.cross_entropy(input=out,target=batch_label) + self.gamma * bit_decay
            prediction = torch.argmax(out,dim=1)
            total_correct += (prediction==batch_label).sum().item()
            total_samples += batch_label.size(0)

        bit_summary = self.model_bit_depth_summary()
        avg_loss , avg_accuracy = total_loss/total_samples , total_correct/total_samples
        return avg_loss, avg_accuracy, bit_summary

    # @torch.compile # this will not work if you want to track modules states in dict and such.. dynamo error. compile model instead.
    def train(self,num_epochs=10):
        pbar_epoch = tqdm(range(num_epochs))
        for epoch in pbar_epoch:
            batch_count = 0
            for batch_img, batch_label in self.train_loader:
                # get new batch [this gets compiled away... this is ok!..]
                batch_img = batch_img.to("cuda").to(self.dtype)
                batch_label = batch_label.to("cuda")

                # bfloat16 with cross entropy needs autocasting 
                # with torch.cuda.amp.autocast(): <- this got deprecated
                with torch.autocast(device_type="cuda",dtype=self.amp_dtype):
                    out = self.model(batch_img)
                    bit_decay = self._qlayersize() / self.qtot_init
                    loss = torch.nn.functional.cross_entropy(input=out,target=batch_label) + self.gamma * bit_decay

                # loss.backward()
                self.scaler.scale(loss).backward()
                # self.optim.step()
                self.scaler.step(self.optim)
                # flush previous grad
                self.optim.zero_grad()
                # remember to update the scaler for next iter
                self.scaler.update()

                # progress bar update [batch count] 
                batch_count +=1
                if batch_count % self.pbar_track_freq == 0:
                    activekernels = self._activekernelscount()
                    activeduals = self._activeLinearDualbasis()
                    pbar_epoch.set_postfix(
                        loss=loss.item(),
                        activekernels = activekernels.values(),
                        activeduals = activeduals.values(),
                        decay=bit_decay.item(),
                    )
                    self._track(loss=loss.item(),
                                activekernels=activekernels,
                                activeduals = activeduals,
                                bit_decay=bit_decay.item(),
                                epoch=epoch)

            # eval tracking
            if epoch % self.eval_track_freq == 0:
                self.model.eval()
                avg_loss , avg_accuracy, bit_summary = self.eval_run()
                print(f"Eval Run Stats {epoch=}, {avg_loss=}, {avg_accuracy=} {bit_summary=}")
                if self.logging:
                    self.experiment.log_metric(name="Eval loss", value=avg_loss, step=epoch)
                    self.experiment.log_metric(name="Eval Accuracy", value=avg_accuracy, step=epoch)
                    self.experiment.log_metric(name="Eval Accuracy", value=avg_accuracy, step=epoch)
                    self.experiment.log_metric(name="Model ULP", value=bit_summary, step=epoch)
        # finishing up
        checkpoint = self._save_checkpoint()
        if self.logging:
            # this also saves the model weights
            log_model(self.experiment,checkpoint,model_name=self.tag)
        torch.save(checkpoint,f"assets/{self.tag}_ckpt.pth_{timestamp}")

        return self.model
    
