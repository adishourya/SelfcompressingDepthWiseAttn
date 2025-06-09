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

mnist = datasets.MNIST(root='./data', train=True, download=True,transform=transform)
train_size = int(0.75 * len(mnist))
eval_size = int(0.1 * len(mnist))
test_size = int(0.15 * len(mnist))
train_dataset, eval_dataset,test_dataset = random_split(mnist, [train_size,eval_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)


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
    print(len(train_loader), train_size) # total training batches
    print(len(eval_loader), eval_size) # total eval batches
    print(len(test_loader), test_size) # total test batches
    show_data_examples(eval_loader,5)
