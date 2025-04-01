import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import CelebADataset, Generator, Critic

# Параметри
batch_size = 128
num_epochs = 20
lr_g = 1e-4
lr_d = 1e-4
beta1, beta2 = 0.0, 0.9  # Рекомендовані для WGAN-GP
nz = 100
d_conv_dim = 64
g_conv_dim = 128
PRINT_EVERY_EPOCH = 1
n_critic = 5
lambda_gp = 10  # штраф для градієнта

data_path = r"/Users/aleksejzarickij/.cache/kagglehub/datasets/jessicali9530/celeba-dataset/versions/2/img_align_celeba/img_align_celeba"

def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = critic(interpolates)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    grad_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((grad_norm - 1) ** 2).mean()
    return gradient_penalty

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor()
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = CelebADataset(root_dir=data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=4, persistent_workers=True)

    generator = Generator(nz, conv_dim=g_conv_dim).to(device)
    critic = Critic(conv_dim=d_conv_dim).to(device)

    optimizerD = optim.Adam(critic.parameters(), lr=lr_d, betas=(beta1, beta2))
    optimizerG = optim.Adam(generator.parameters(), lr=lr_g, betas=(beta1, beta2))

    fixed_noise = torch.randn(64, nz, device=device)
    G_losses, D_losses = [], []

    for epoch in range(num_epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for i, data in enumerate(loop):
            real = data.to(device)
            b_size = real.size(0)

            for _ in range(n_critic):
                noise = torch.randn(b_size, nz, device=device)
                fake = generator(noise).detach()
                output_real = critic(real).mean()
                output_fake = critic(fake).mean()

                gp = compute_gradient_penalty(critic, real, fake, device)
                lossD = -(output_real - output_fake) + lambda_gp * gp

                optimizerD.zero_grad()
                lossD.backward()
                optimizerD.step()

            noise = torch.randn(b_size, nz, device=device)
            fake = generator(noise)
            output_fake_for_G = critic(fake).mean()
            lossG = -output_fake_for_G

            optimizerG.zero_grad()
            lossG.backward()
            optimizerG.step()

            G_losses.append(lossG.item())
            D_losses.append(lossD.item())
            loop.set_postfix(loss_D=lossD.item(), loss_G=lossG.item())

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss_D: {lossD.item():.4f} Loss_G: {lossG.item():.4f}")

        if (epoch + 1) % PRINT_EVERY_EPOCH == 0:
            with torch.no_grad():
                fake_images = generator(fixed_noise).detach().cpu()
            grid = utils.make_grid(fake_images, padding=2, normalize=True)
            plt.figure(figsize=(8, 8))
            plt.axis("off")
            plt.title(f"Generated Images at Epoch {epoch + 1}")
            plt.imshow(np.transpose(grid, (1, 2, 0)))
            plt.show()

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Critic Loss During Training")
    plt.plot(G_losses, label="G Loss")
    plt.plot(D_losses, label="Critic Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
