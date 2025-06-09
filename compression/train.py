import torch
import cutlass.cute as cute
assert torch.cuda.is_available() , "Not optimized for any other backend"
print(torch.__version__)

from data.mnist_data import train_loader, eval_loader
from qmodules.Models.ConvModel import QconvModel
from QTrainer import QTrainer


if __name__ == "__main__":
    qtrainer = QTrainer(model=QconvModel,
                        train_loader=train_loader,
                        eval_loader=eval_loader,
                        logging=False)
    qtrainer.train(1)
