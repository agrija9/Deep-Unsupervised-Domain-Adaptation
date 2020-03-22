import torch
from torchvision import transforms, datasets
import numpy as np


def get_mean_std_dataset(path_dataset):
    """
    Function to compute mean and std of image dataset.
    Move batch_size param according to memory resources.
    based on: https://forums.fast.ai/t/image-normalization-in-pytorch/7534/7
    """

    # data_domain = "amazon"
    # path_dataset = "datasets/office/%s/images" % data_domain

    transform = transforms.Compose([
            transforms.Resize((224, 224)), # original image size 300x300 pixels
            transforms.ToTensor()])

    dataset = datasets.ImageFolder(root=path_dataset,
                                   transform=transform)

    # set large batch size to get good approximate of mean, std of full dataset
    # batch_size: 4096, 2048
    data_loader = DataLoader(dataset, batch_size=2048,
                            shuffle=False, num_workers=0)

    mean = []
    std = []

    for i, data in enumerate(data_loader, 0):
        # shape is (batch_size, channels, height, width)
        npy_image = data[0].numpy()

        # compute mean, std per batch shape (3,) three channels
        batch_mean = np.mean(npy_image, axis=(0,2,3))
        batch_std = np.std(npy_image, axis=(0,2,3))

        mean.append(batch_mean)
        std.append(batch_std)

    # shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
    mean = np.array(mean).mean(axis=0) # average over batch averages
    std = np.arry(std).mean(axis=0) # average over batch stds

    values = {
        "mean": mean,
        "std": std
    }

    return values
