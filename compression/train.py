import torch
import cutlass.cute as cute
assert torch.cuda.is_available() , "Not optimized for any other backend"
print(torch.__version__)

from data.mnist_data import train_loader, eval_loader
from qmodules.Models.ConvModel import QconvModel
from QTrainer import QTrainer

config = dict(
    pbar_track_freq=50, # Every xth batch updates the progress bar
    eval_track_freq= 5, # Every xth epoch does an Eval Run 
    logging=True, # comet ml tracking 
    comet_username="adishourya",
    tag="low_bval",
    project_name="convolution_compressing"
)


if __name__ == "__main__":
    qtrainer = QTrainer(model=QconvModel,
                        train_loader=train_loader,
                        eval_loader=eval_loader,
                        **config)
    qtrainer.train(200)
