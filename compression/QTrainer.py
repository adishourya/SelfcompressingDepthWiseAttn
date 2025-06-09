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
                 logging=False,
                 tag="low_bval",
                 project_name="convolution_compressing"):

        self.model = model()
        self.train_loader =train_loader
        self.eval_loader = eval_loader

        assert isinstance(dtype,torch.dtype),"Hein? unrecognized dtype"
        self.dtype = dtype
        self.logging = logging

        if self.logging:
            # print("logging costs around 0.5 secs more per iteration.")
            self.tag = tag
            self.project_name = project_name
            self._setup_logging()
        

        # does not work on my machine with less precision...
        if self.dtype == torch.float32:
            # do we actually need high precision matmul ??
            torch.set_float32_matmul_precision("high")

        print(f"Training with {self.dtype=} precision")

        # self.scaler = torch.amp.GradScaler()
        self.model = self.model.to(self.dtype)
        self.model = torch.compile(self.model)

        # dont try it on cpu!
        self.model.to("cuda")
        self.track_decay = []
        self.track_activekernels = []
        self.track_loss = []

    
        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            weight_decay=1e-3)
    
        self.gamma = (1/10) # should be around 0.05 or something.. compression factor
        # we need to calculate total number of parameters at initialization (papar calls it N)
        # here since everything is trainable

        self.tot_init = sum(p.numel() for group in self.optim.param_groups for p in group['params'] if p.requires_grad)
        self.tot_qparams = torch.sum(torch.tensor([p_weight.numel() for p,p_weight in self.model.named_parameters() if "_bit" in p]))
    
        print(f"Total Parameters {self.tot_init=}")
        print(f"of which compression are :{self.tot_qparams=}")
        print(f"compression factor at init {self.gamma * self._qlayersize()}")
    
        # print(self._qlayersize())
        print("Total Kernels:",self._activekernelscount())

    def _setup_logging(self):
        import os
        os.getenv("comet_api")
        
        from comet_ml import start
        from comet_ml.integration.pytorch import log_model,watch
        
        self.experiment = start(
          api_key=os.getenv("comet_api"),
          project_name=self.project_name,
          workspace="adishourya"
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
        return (size_conv + size_lin)/self.tot_init


    def _activekernelscount(self):
        kernel_counts = dict()
        for name,layer in self.model.named_modules():
            if isinstance(layer,Qconv):
                depths = torch.relu(layer.depth_bit)
                count =torch.sum(torch.where(depths>0,1,0)).item()
                kernel_counts[name] = count
        return kernel_counts


    def _track(self,loss,activekernels,bit_decay,epoch):
        self.track_loss.append(loss)
        self.track_activekernels.append(activekernels.values())
        self.track_decay.append(bit_decay)
        if self.logging:
             self.experiment.log_metric(name="loss", value=loss, step=epoch)
             self.experiment.log_metric(name="decay", value=bit_decay, step=epoch)
             self.experiment.log_metric(name="activekernels", value=sum(activekernels.values()), step=epoch)
        

    # @torch.compile # this will not work if you want to track modules states in dict and such.. dynamo error. compile model instead.
    def train(self,num_epochs=10):
        pbar_epoch = tqdm(range(num_epochs))
        for epoch in pbar_epoch:
            for batch_img, batch_label in self.train_loader:
                batch_img = batch_img.to("cuda").to(self.dtype)
                batch_label = batch_label.to("cuda")

                # bfloat16 with cross entropy needs autocasting 
                with torch.autocast(device_type="cuda",dtype=self.dtype):
                    out = self.model(batch_img)
                    bit_decay = self._qlayersize()
                    loss = torch.nn.functional.cross_entropy(input=out,target=batch_label) + self.gamma * bit_decay

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                if epoch %50 == 0:
                    activekernels = self._activekernelscount()
                    pbar_epoch.set_postfix(
                        loss=loss.item(),
                        activekernels = activekernels.values(),
                        decay=self.gamma*bit_decay.item(),
                    )
                    self._track(loss.item(),activekernels,self.gamma*bit_decay.item(),epoch)
                        
        if self.logging:
            log_model(self.experiment,model=self.model,model_name=self.tag)
        return self.model
    
