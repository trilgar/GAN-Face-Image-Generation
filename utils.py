import os

import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


# Датасет
class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None, cache_in_memory=False):
        self.root_dir = root_dir
        self.image_files = [os.path.join(root_dir, f)
                            for f in os.listdir(root_dir)
                            if f.lower().endswith(('.jpg', '.png'))]
        self.transform = transform
        self.cache_in_memory = cache_in_memory

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


# Генератор
class Generator(nn.Module):
    def __init__(self, nz, ngf, nc, image_size):
        super(Generator, self).__init__()
        self.ngf = ngf
        self.main = nn.Sequential(
            nn.Linear(nz, ngf * 8 * 7 * 6),
            nn.BatchNorm1d(ngf * 8 * 7 * 6),
            nn.ReLU(True)
        )
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),  # -> (ngf*4, 14, 12)
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),  # -> (ngf*2, 28, 24)
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),  # -> (ngf, 56, 48)
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, 32, kernel_size=4, stride=2, padding=1, bias=False),  # -> (32, 112, 96)
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, nc, kernel_size=4, stride=2, padding=1, bias=False),  # -> (nc, 224, 192)
            nn.Tanh()  # [-1, 1]
        )
        # Адаптуємо фінальний розмір до (218, 178)
        self.adapt = nn.AdaptiveAvgPool2d(image_size)

    def forward(self, input):
        x = self.main(input)  # (batch_size, ngf*8*7*6)
        x = x.view(-1, self.ngf * 8, 7, 6)
        x = self.conv_blocks(x)  # (batch_size, nc, 224, 192)
        x = self.adapt(x)  # (batch_size, nc, 218, 178)
        return x


# Дискримінатор
class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Вхід: (nc, 218, 178)
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),  # -> (ndf, 109, 89)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),  # -> (ndf*2, ~55, ~45)
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),  # -> (ndf*4, ~27, ~23)
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),  # -> (ndf*8, ~13, ~12)
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # Остання згортка, яка зводить просторовий розмір до 1x1
            nn.Conv2d(ndf * 8, 1, kernel_size=(13, 11), stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(input.size(0))
