import torch
import cutlass.cute as cute
assert torch.cuda.is_available() , "Not optimized for any other backend"
print(torch.__version__)

data_config = dict(
        train_split = 0.75,
        eval_split = 0.1,
        test_split = 0.15,
        batch_size = 256
)
from data.mnist_data import get_dataloader
train_loader,eval_loader,_ = get_dataloader(**data_config)


from qmodules.Models.ConvModel import QconvModel
from qmodules.Models.PureConvModel import QPureconvModel
from QTrainer import QTrainer

train_config = dict(
    # model = QPureconvModel, 
    model = QconvModel, 
    train_loader = train_loader,
    eval_loader = eval_loader,
    dtype = torch.float32,      # overflows if natively trained at fp16
    amp_dtype = torch.bfloat16, # "simulate" automatic mixed precision type 
    compression_gamma = 0.1, # layersize coefficient
    pbar_track_freq=50, # Every xth batch updates the progress bar
    eval_track_freq= 5, # Every xth epoch does an Eval Run 
    logging=True, # comet ml tracking 
    comet_username="adishourya",
    tag="pure_conv",
    project_name="convolution_compressing"
)


if __name__ == "__main__":
    qtrainer = QTrainer(**train_config)
    # qtrainer.train(1)
    # qtrainer.train(300)
    qtrainer.train(1000)
