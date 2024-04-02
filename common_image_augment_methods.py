from PIL import Image
import torch
import torchvision
from torch import nn

from matplotlib import pyplot as plt
from utils.util import set_figsize
from utils.util import show_images

def apply(img, aug, num_rows=2, num_cols=4, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    show_images(Y, num_rows, num_cols, scale=scale)

def main():
    set_figsize()
    img = Image.open('img/cat1.jpg')
    #plt.imshow(img)
    
    # Flipping 
    # apply(img, torchvision.transforms.RandomHorizontalFlip())
    # apply(img, torchvision.transforms.RandomVerticalFlip())

    # resizing and Cropping
    shape_aug = torchvision.transforms.RandomResizedCrop(
        (200, 200), scale=(0.1, 1), ratio=(0.5, 2))
    # apply(img, shape_aug)

    # # Changing Colors,  four aspects of the image color: brightness, contrast, saturation, and hue
    # apply(img, torchvision.transforms.ColorJitter(brightness=0.5, 
    #     contrast=0, saturation=0, hue=0))
    # apply(img, torchvision.transforms.ColorJitter(brightness=0, 
    #     contrast=0, saturation=0, hue=0.5))
    
    color_aug = torchvision.transforms.ColorJitter(
        brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)

    # combine multiple image augmentation methods
    augs = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(), 
        color_aug, shape_aug])
    apply(img, augs)
    plt.show()


if __name__ == '__main__':
    main()

