import os
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import torch.nn as nn

from models import Generator, Discriminator

def train():
    batch_size = 64  # Smanjen batch_size za stabilnost
    epochs = 100
    latent_dim = 100
    lr = 0.0001
    b1 = 0.5
    b2 = 0.999
    num_classes = 10

    cuda = True if torch.cuda.is_available() else False
    device = torch.device('cuda' if cuda else 'cpu')
    print(f"Using device: {device}")

    os.makedirs("images", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)

    transformacije = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = datasets.MNIST("mnist_data", train=True, download=True, transform=transformacije)
    img_size = dataset[0][0].shape[1]
    channels = dataset[0][0].shape[0]

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    adversarial_loss = nn.BCELoss()

    generator = Generator(img_size, latent_dim, channels, num_classes).to(device)
    discriminator = Discriminator(img_size, channels, num_classes).to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    for epoch in range(epochs):
        loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=True)
        for i, (imgs, labels) in loop:
            current_batch_size = imgs.size(0)

            # Kreiranje tensora za valid/fake
            valid = torch.ones(current_batch_size, 1, device=device, requires_grad=False)
            fake = torch.zeros(current_batch_size, 1, device=device, requires_grad=False)

            real_imgs = imgs.to(device)
            labels = labels.to(device)

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(current_batch_size, latent_dim, device=device)
            gen_labels = torch.randint(0, num_classes, (current_batch_size,), device=device)
            gen_imgs = generator(z, gen_labels)
            g_loss = adversarial_loss(discriminator(gen_imgs, gen_labels), valid)
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs, labels), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), gen_labels), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(d_loss=d_loss.item(), g_loss=g_loss.item())

            batches_done = epoch * len(dataloader) + i
            if batches_done % 200 == 0:
                save_image(gen_imgs.data[:25], f"images/{batches_done}.png", nrow=5, normalize=True)

        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict()
        }, f'models/checkpoints/chk{epoch}.pth')

if __name__ == '__main__':
    train()
