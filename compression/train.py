import torch
import cutlass.cute as cute
assert torch.cuda.is_available() , "Not optimized for any other backend"
print(torch.__version__)

# this should be done automatically from pytorch >=2.5
from einops._torch_specific import allow_ops_in_compiled_graph
allow_ops_in_compiled_graph()

data_config = dict(
        train_split = 0.75,
        eval_split = 0.1,
        test_split = 0.15,
        batch_size = 256
)
from data.mnist_data import get_dataloader as mnist_loader
from data.country_data import get_dataloader as country_loader

train_loader,eval_loader,_ = mnist_loader(**data_config)
# train_loader,eval_loader,_ = country_loader(batch_size=32)


from qmodules.Models.ConvModel import QconvModel
from qmodules.Models.PureConvModel import QPureconvModel
from qmodules.Models.eagerDWLin import EagerDWLin 
from QTrainer import QTrainer

train_config = dict(
    # model = QPureconvModel(), 
    model = EagerDWLin(img_channels=1,num_heads=16,head_dim=16,repeat_transformer=10,num_classes=10), 
    # model = EagerDWLin(img_channels=1,num_heads=16,head_dim=16,repeat_transformer=18,num_classes=10), 
    # model = QconvModel(),
    train_loader = train_loader,
    eval_loader = eval_loader,
    to_compile = True,
    dtype = torch.float32,      # overflows if natively trained at fp16
    amp_dtype = torch.bfloat16, # "simulate" automatic mixed precision type 
    compression_gamma = 0.8, # layersize coefficient
    pbar_track_freq=50, # Every xth batch updates the progress bar
    eval_track_freq= 5, # Every xth epoch does an Eval Run 
    logging=True, # comet ml tracking 
    comet_username="adishourya",
    tag="DWAttn",
    project_name="convolution_compressing"
)

if __name__ == "__main__":
    qtrainer = QTrainer(**train_config)
    # qtrainer.train(1)
    qtrainer.train(20)
    # qtrainer.train(1000)
    # qtrainer.train(100)
