import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, img_size, latent_dim, channels, num_classes):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.init_size = img_size // 4
        self.latent_dim = latent_dim
        self.channels = channels
        self.num_classes = num_classes

        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.l1 = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128 * self.init_size ** 2)
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        label_input = self.label_emb(labels)
        gen_input = torch.cat((noise, label_input), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_size, channels, num_classes):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.channels = channels
        self.num_classes = num_classes

        self.label_emb = nn.Embedding(num_classes, num_classes)

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        # Channels + 1 za label kao dodatni kanal
        self.model = nn.Sequential(
            *discriminator_block(channels + 1, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # Koristi LazyLinear za automatsko određivanje dimenzija
        self.adv_layer = nn.Sequential(
            nn.LazyLinear(1), 
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Embedding labela
        label_input = self.label_emb(labels)
        
        # Proširuj label na dimenzije slike
        label_map = label_input.view(labels.size(0), self.num_classes, 1, 1)
        label_map = label_map.expand(labels.size(0), self.num_classes, self.img_size, self.img_size)
        
        # Uzmi samo prvi kanal od label_map i spoji sa slikom
        img_input = torch.cat((img, label_map[:, :1, :, :]), 1)
        
        # Konvolucijski slojevi
        out = self.model(img_input)
        
        # Flatten
        out = out.view(out.shape[0], -1)
        
        # Finalni sloj - LazyLinear će automatski odrediti dimenzije
        validity = self.adv_layer(out)
        return validity
