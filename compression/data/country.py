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

# this normalization was recommended in imagenet
transform = transforms.Compose([
    transforms.Resize((224, 224)),# r244
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),  # R, G, B means
        std=(0.229, 0.224, 0.225)    # R, G, B stds
    )
])

def get_dataloader(batch_size=256):
    # country211 is about 11Gigs
    country_train = datasets.Country211(root=data_dir,split="train",download=True,transform=transform)
    country_valid= datasets.Country211(root=data_dir,split="valid",download=True,transform=transform)
    country_test= datasets.Country211(root=data_dir,split="test",download=True,transform=transform)
    # then make
    train_loader = DataLoader(country_train, batch_size=batch_size, shuffle=True, num_workers=2)
    eval_loader = DataLoader(country_valid, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(country_test, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader , eval_loader , test_loader


def show_data_examples(loader,count=5):
    img , labels = next(iter(loader)) 
    print(f"Shapes: ",img.shape, labels.shape)
    eg,eg_label = img[:count] , labels[:count]
    eg = einops.rearrange(eg,"b c h w -> h (b w) c")
    plt.suptitle("Country 211")
    plt.title(eg_label)
    plt.imshow(eg)
    plt.show()
    print(eg_label)


if __name__ == "__main__":
    train_loader , eval_loader , test_loader = get_dataloader()
    print(len(train_loader)) # total training batches
    print(len(eval_loader)) # total eval batches
    print(len(test_loader)) # total test batches
    show_data_examples(eval_loader,5)
