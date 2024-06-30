import os
import imageio
from tqdm.auto import tqdm
# Create GIF
# Duration = Total_Images // FPS

images = []
for file_name in tqdm(sorted(os.listdir("runs/fake/saves"))):
    images.append(imageio.imread(os.path.join('runs/fake/saves', file_name)))
imageio.mimsave('gan_fake.gif', images, fps=12)

print("GIF created and saved as 'gan_fake.gif'")