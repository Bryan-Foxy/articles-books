import time
import imageio
import models
import torch 
import torchvision
import numpy as np 
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

# Params
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
bs = 32
latent_size = 128
hidden_size_gen = 256
hidden_size_disc = 128
img_size = 28 * 28 * 1 # We are using the dataset MNIST
criterion = torch.nn.BCELoss()
num_epochs = 200
f_noise = torch.randn((bs,latent_size)).to(device)
step = 0
writer_tensorboard = SummaryWriter(f"runs/real")
writer_tensorboard_fake = SummaryWriter(f"runs/fake")

# Datasets
transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean = (0.5,), std = (0.5,))
]
)

dset = torchvision.datasets.MNIST(
    root = "data/",
    train = True,
    transform = transforms,
    download = True
)

loader = torch.utils.data.DataLoader(
    dset,
    batch_size = bs,
    shuffle = True
)


generator = models.Gen(latent_size, hidden_size_gen, img_size).to(device)
discriminator = models.Disc(img_size, hidden_size_disc).to(device)
optimizer_gen = torch.optim.Adam(generator.parameters(), lr=lr)
optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=lr)
start = time.time()
for epoch in tqdm(range(num_epochs)):
    for batch_idx, (data, _) in enumerate(loader):
        data = data.view(-1,img_size).to(device) # Flatten image
        batch_size = data.shape[0]
        noise = torch.randn(batch_size, latent_size).to(device) # Adding nosie
        fake_data = generator(noise)
        disc_data = discriminator(data).view(-1)
        loss_disc_data = criterion(disc_data, torch.ones_like(disc_data))
        disc_fake = discriminator(fake_data).view(-1)
        loss_disc_fake_data = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_data + loss_disc_fake_data) / 2
        discriminator.zero_grad()
        loss_disc.backward(retain_graph = True)
        optimizer_disc.step()

        out = discriminator(fake_data).view(-1)
        loss_gen = criterion(out, torch.ones_like(out))
        generator.zero_grad()
        loss_gen.backward()
        optimizer_gen.step()
    
    print(f"Epochs [{epoch+1}/{num_epochs}]: \t Loss of the Generator: {loss_gen:.3f} | Loss of the Discriminator: {loss_disc:.3f}")
    with torch.no_grad():
        fake_img = generator(f_noise).reshape(-1,1,28,28)
        img = data.reshape(-1,1,28,28)
        img_make_grid_fake = torchvision.utils.make_grid(fake_img, normalize=True)

        img_np = img_make_grid_fake.cpu().numpy().transpose(1,2,0)
        imageio.imwrite(f"runs/fake/saves/epoch_{epoch+1}.png", (img_np*255).astype(np.uint8))

        img_make_grid = torchvision.utils.make_grid(img, normalize=True)
        
        writer_tensorboard_fake.add_image(
            "MNIST Fake images",
            img_make_grid_fake,
            global_step = step
        ) 

        writer_tensorboard.add_image(
            "MNIST Real images",
            img_make_grid,
            global_step = step
        )

        step += 1





        



    