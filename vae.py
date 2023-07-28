# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image

from random import randint

# %%
# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# %%
batch_size = 64

# %%
# Load data
dataset = ImageFolder(
    root="../../dataset/",
    transform=transforms.Compose([transforms.Grayscale(), transforms.Resize((64, 64)), transforms.ToTensor()]),
)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(len(dataset.imgs), len(dataloader))

# %%
# Fixed input for debugging
fixed_x, _ = next(iter(dataloader))
save_image(fixed_x, "real_image.png")

# Image('real_images.png')

# %%
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


# %%
class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)


# %%
class VAE(nn.Module):
    def __init__(self, image_channels=1, h_dim=1024, z_dim=32):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten(),
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar


# %%
image_channels = fixed_x.size(1)

# %%
vae = VAE(image_channels=image_channels).to(device)
# model.load_state_dict(torch.load('vae.torch', map_location='cpu'))

# %%
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

# %%
def loss_fn(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x.to(device), x.to(device), size_average=False)
    # BCE = F.mse_loss(recon_x, x, size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD, BCE, KLD


# %%
# !rm -rfr reconstructed
# !mkdir reconstructed

# %%
epochs = 20

# %%
for epoch in range(epochs):
    for idx, (images, _) in enumerate(dataloader):
        recon_images, mu, logvar = vae(images.to(device))
        loss, bce, kld = loss_fn(recon_images, images, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(
            "Epoch[{}/{}] Loss: {:.3f} {:.3f} {:.3f}".format(
                epoch + 1,
                epochs,
                loss.item() / batch_size,
                bce.item() / batch_size,
                kld.item() / batch_size,
            )
        )

# # notify to android when finished training
# notify(to_print, priority=1)

torch.save(vae, "vae.pth")

# %%
def compare(x):
    recon_x, _, _ = vae(x)
    return torch.cat([x, recon_x])


# %%
fixed_x = dataset[randint(1, 10)][0].unsqueeze(0)
compare_x = compare(fixed_x.to(device))

save_image(compare_x.data.cpu(), "sample_image.png")
# display(Image('sample_image.png', width=700, unconfined=True))
