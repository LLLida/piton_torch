# Simple denoiser using CNNs
# performance: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
# https://pytorch.org/tutorials/beginner/saving_loading_models.html
# PyTorch tutorial: https://github.com/udacity/deep-learning-v2-pytorch/blob/master/batch-norm/Batch_Normalization.ipynb
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class ConvDenoiser(torch.nn.Module):
    def __init__(self):
        super(ConvDenoiser, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.t_conv1 = nn.ConvTranspose2d(8, 8, 3, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(8, 16, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(16, 32, 2, stride=2)
        self.conv_out = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))

        x = F.sigmoid(self.conv_out(x))

        return x

model = ConvDenoiser()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0015)

# load the training and test datasets
train_data = datasets.MNIST(root='~/.pytorch/MNIST_data/', train=True,
                                   download=False, transform=ToTensor())
test_data = datasets.MNIST(root='~/.pytorch/MNIST_data/', train=False,
                                  download=False, transform=ToTensor())

noise_factor = 0.3

num_workers = 2
batch_size = 20

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

mode = int(input("Enter the mode(0 - training, 1 - testing): "))

model_path = "denoiser.bin"

print(torch.__config__.show())
print(torch.__config__.parallel_info())

if mode == 0:
    epochs = int(input("Enter the number of epochs: "))
    print('Starting the training process...')
    for epoch in range(epochs+1):
        train_loss = 0.0

        for data in train_loader:
            images, _ = data

            noisy_imgs = images + noise_factor * torch.randn(*images.shape)
            noisy_imgs = np.clip(noisy_imgs, 0.0, 1.0)

            optimizer.zero_grad()
            outputs = model.forward(noisy_imgs)

            loss = criterion(outputs, images)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader)
        print(f'Epoch: {epoch}\tTraining loss: {train_loss:.6f}')
    torch.save(model.state_dict(), model_path)

elif mode == 1:
    model.load_state_dict(torch.load(model_path))
    model.eval()

    dataiter = list(iter(test_loader))
    images, labels = dataiter[0]

    noisy_imgs = np.clip(images + noise_factor * torch.randn(*images.shape), 0., 1.)

    output = model(noisy_imgs)
    noisy_imgs = noisy_imgs.numpy()

    # output is resized into a batch of images
    output = output.view(batch_size, 1, 28, 28)
    # use detach when it's an output that requires_grad
    output = output.detach().numpy()

    # plot the first ten input images and then reconstructed images
    fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25,4))

    # input images on top row, reconstructions on bottom
    for noisy_imgs, row in zip([noisy_imgs, output], axes):
        for img, ax in zip(noisy_imgs, row):
            ax.imshow(np.squeeze(img), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.show()
