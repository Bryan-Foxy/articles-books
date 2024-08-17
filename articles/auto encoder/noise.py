import numpy as np 
from tqdm import tqdm
from skimage.util import random_noise

def add_noise(data):
    """
    """
    new_data = np.copy(data)
    for i, image in tqdm(enumerate(new_data)):
        image=random_noise(image, mode='gaussian', mean=0, var=0.3)
        image=random_noise(image, mode='s&p', amount=0.2, salt_vs_pepper=0.5)
        image=random_noise(image, mode='poisson')
        image=random_noise(image, mode='speckle', mean=0, var=0.1)
        new_data[i]=image
    print("Noise adding sucessfully.")
    return new_data
