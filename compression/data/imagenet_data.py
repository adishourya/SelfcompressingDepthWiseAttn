from pathlib import Path
script_dir = Path(__file__).resolve().parent
data_dir = script_dir / "datasets"  # Keep datasets in a "datasets" folder next to this file

import matplotlib.pyplot as plt
import einops
import torch
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
torch.manual_seed(10)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def get_dataloader(train_split:float = 0.75,
                   eval_split:float = 0.1,
                   test_split:float =0.15,
                   batch_size=256):

    assert train_split + eval_split + test_split == 1 , "split does not sum to 1"
    mnist = datasets.ImageNet(root=data_dir, train=True, download=True,transform=transform)
    train_size = int(train_split * len(mnist))
    eval_size = int(eval_split * len(mnist))
    test_size = int(test_split * len(mnist))
    train_dataset, eval_dataset,test_dataset = random_split(mnist, [train_size,eval_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader , eval_loader , test_loader


def show_data_examples(loader,count=5):
    loader = iter(loader)
    img , labels = next(loader) 
    eg,eg_label = img[:count] , labels[:count]
    eg = einops.rearrange(eg,"b c h w -> h (b w) c")
    plt.title(eg_label)
    plt.imshow(eg,cmap="Blues")
    plt.show()
    print(eg_label)


if __name__ == "__main__":
    train_loader , eval_loader , test_loader = get_dataloader()
    print(len(train_loader)) # total training batches
    print(len(eval_loader)) # total eval batches
    print(len(test_loader)) # total test batches
    show_data_examples(eval_loader,5)
