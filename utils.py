import os

import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import torch.nn.functional as F


# Датасет
class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None, cache_in_memory=False, max_images=None):
        self.root_dir = root_dir
        self.image_files = [os.path.join(root_dir, f)
                            for f in os.listdir(root_dir)
                            if f.lower().endswith(('.jpg', '.png'))]
        self.transform = transform
        self.cache_in_memory = cache_in_memory
        self.max_images = max_images

        # Якщо cache_in_memory встановлено, завантажуємо всі зображення в RAM
        if self.cache_in_memory:
            self.cached_data = []
            print("Caching images in memory...")
            for img_path in tqdm(self.image_files):
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                self.cached_data.append(image)
            print("Caching complete.")

    def __len__(self):
        if self.max_images:
            return self.max_images
        else:
            return len(self.image_files)

    def __getitem__(self, idx):
        if self.cache_in_memory:
            return self.cached_data[idx]
        else:
            img_path = self.image_files[idx]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image


def deconv(in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True):
    layers = []

    # append transpose conv layer -- we are not using bias terms in conv layers
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding))

    # optional batch norm layer
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)


# Генератор
class Generator(nn.Module):

    def __init__(self, z_size, conv_dim):
        super(Generator, self).__init__()
        self.conv_dim = conv_dim

        self.fc = nn.Linear(z_size, conv_dim * 4 * 4 * 4)
        # complete init function

        self.de_conv1 = deconv(conv_dim * 4, conv_dim * 2)
        self.de_conv2 = deconv(conv_dim * 2, conv_dim)
        self.de_conv3 = deconv(conv_dim, 3, 4, batch_norm=False)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network
        :return: A 32x32x3 Tensor image as output
        """
        # define feedforward behavior
        x = self.fc(x)
        x = self.dropout(x)

        x = x.view(-1, self.conv_dim * 4, 4, 4)

        x = F.relu(self.de_conv1(x))
        x = F.relu(self.de_conv2(x))
        x = self.de_conv3(x)
        x = F.tanh(x)

        return x


def conv(in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True):
    layers = []
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                           kernel_size=kernel_size, stride=stride, padding=padding)
    # appending convolutional layer
    layers.append(conv_layer)
    # appending batch norm layer
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    return nn.Sequential(*layers)


class Discriminator(nn.Module):

    def __init__(self, conv_dim):
        super(Discriminator, self).__init__()

        # complete init function
        self.conv_dim = conv_dim

        self.conv1 = conv(3, conv_dim, batch_norm=False)
        self.conv2 = conv(conv_dim, conv_dim * 2)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4)
        self.conv4 = conv(conv_dim * 4, conv_dim * 8)
        self.fc = nn.Linear(conv_dim * 4 * 4 * 2, 1)

    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network
        :return: Discriminator logits; the output of the neural network
        """
        # define feedforward behavior
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)

        x = x.view(-1, self.conv_dim * 4 * 2 * 4)

        x = self.fc(x)

        return x.squeeze()
