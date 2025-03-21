import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from tqdm import tqdm

from utils import CelebADataset, Generator, Discriminator

# Параметри
batch_size = 128
num_epochs = 25
lr = 2e-3
beta1 = 0.5
PRINT_EVERY_EPOCH = 1

image_size = (218, 178)  # (висота, ширина)
nc = 3  # кількість каналів (RGB)
nz = 100  # розмір латентного вектора
ngf = 64  # базова кількість фільтрів у генераторі
ndf = 64  # базова кількість фільтрів у дискримінаторі

data_path = r"C:\Users\zarit\.cache\kagglehub\datasets\jessicali9530\celeba-dataset\versions\2\img_align_celeba\img_align_celeba"

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    # Пристрій
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = CelebADataset(root_dir=data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
                            pin_memory_device="cuda", prefetch_factor=4)


    # Ініціалізація ваг
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


    generator = Generator(nz, ngf, nc, image_size).to(device)
    generator.apply(weights_init)

    discriminator = Discriminator(nc, ndf).to(device)
    discriminator.apply(weights_init)

    criterion = nn.BCELoss()
    optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

    fixed_noise = torch.randn(64, nz, device=device)

    G_losses = []
    D_losses = []

    # Головний цикл навчання
    for epoch in range(num_epochs):
        for i, data in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            # 1. Оновлення дискримінатора: максимізуємо log(D(x)) + log(1 - D(G(z)))
            discriminator.zero_grad()
            real = data.to(device)
            b_size = real.size(0)
            label = torch.full((b_size,), 1.0, device=device)

            # Прямий прохід для реальних зображень
            output = discriminator(real)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Генерація фейкових зображень
            noise = torch.randn(b_size, nz, device=device)
            fake = generator(noise)
            label.fill_(0.0)
            output = discriminator(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            errD = errD_real + errD_fake
            optimizerD.step()

            # 2. Оновлення генератора: максимізуємо log(D(G(z)))
            generator.zero_grad()
            label.fill_(1.0)
            # Дискримінатор прагне мінімізувати цю втрату, тобто максимізувати log(1 - D(G(z)))
            # іншими словами ми хочемо щоб дискриимнатор вважав згенеровані зображення реальними (1)
            output = discriminator(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Запам'ятовуємо для графіку
            G_losses.append(errG.item())
            D_losses.append(errD.item())

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}")

        # Кожні 5 епох виводимо приклад згенерованих зображень
        if (epoch + 1) % PRINT_EVERY_EPOCH == 0:
            with torch.no_grad():
                fake_images = generator(fixed_noise).detach().cpu()
            grid = utils.make_grid(fake_images, padding=2, normalize=True)
            plt.figure(figsize=(8, 8))
            plt.axis("off")
            plt.title(f"Generated Images at Epoch {epoch + 1}")
            plt.imshow(np.transpose(grid, (1, 2, 0)))
            plt.show()

    # Побудова графіку втрат генератора та дискримінатора на одному plot
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G Loss")
    plt.plot(D_losses, label="D Loss")
    plt.xlabel("Ітерації")
    plt.ylabel("Втрата")
    plt.legend()
    plt.show()
