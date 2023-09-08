"""
Heekyung Kim
CS5330 SP 23
Final Project
This script is used to test and visualize feature extraction.
This script uses convolution layer 11 of SSD as feature extractor.
It then feeds the feature extracted images to convolution layers of mobilenet to classify/detect objects.
"""
import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms, utils
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.detection.ssd import SSD
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
import cv2 as cv

from PIL import Image

# Load model
def load_model(model_fname):
    model = SSD(torch.load(model_fname))
    return model


# Plot filter convolved images
def plot_convolved_images(convolved_images, layer):

    merged_list = []
    for i in range(len(layer)):
        merged_list.append(layer[i][0].detach().numpy())
        merged_list.append(convolved_images[i])

    fig = plt.figure()
    for i in range(len(merged_list[41:63])):
        plt.subplot(6, 4, i + 1)
        plt.tight_layout()
        plt.imshow(merged_list[i], interpolation='none')
        plt.xticks([])
        plt.yticks([])
    plt.show()


# Convolve filter
def conv_filter(data, layer):
    convolved_images = []

    conv_img = cv.filter2D(src=data.numpy(), ddepth=-1, kernel=filter.detach().numpy())

    return conv_img

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def main():
    fname = "./dataset/student_center_set1/test/pause/pause_1.png"
    network = torch.jit.load('model_test1.pt', _restore_shapes= True)

    weight11 = network.state_dict()['backbone.features.0.11.block.0.0.weight']

    image = Image.open(str(fname))

    transform = transforms.ToTensor()
    tensor = transform(image)

    network.eval()
    _, detection = network([tensor])

    weight11 = network.state_dict()['backbone.features.0.11.block.0.0.weight']
    print(weight11.shape) # print the weights to see where the features are highlighted

    plt.imshow( weight11 )
    plt.show()


if __name__ == "__main__":
    main()