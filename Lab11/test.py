import numpy as np
from torchvision.utils import save_image
import torch
from models import Generator

def test():
    batch_size = 64
    latent_dim = 100
    img_size = 28
    channels = 1
    num_classes = 10
    model_to_load = "models/checkpoints/chk99.pth"

    cuda = True if torch.cuda.is_available() else False
    device = torch.device('cuda' if cuda else 'cpu')

    generator = Generator(img_size, latent_dim, channels, num_classes).to(device)
    checkpoint = torch.load(model_to_load, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()

    with torch.no_grad():
        for digit in range(num_classes):
            z = torch.randn(batch_size, latent_dim, device=device)
            labels = torch.full((batch_size,), digit, device=device, dtype=torch.long)
            gen_imgs = generator(z, labels)
            save_image(gen_imgs.data, f"images/generated_digit_{digit}.png", nrow=8, normalize=True)

if __name__ == '__main__':
    test()
