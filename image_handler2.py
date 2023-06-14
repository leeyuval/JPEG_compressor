import numpy as np
import imageio.v2 as imageio
import scipy
from PIL import Image
import matplotlib.pyplot as plt
import os

from dct import *
from quantizer import *
from image_operations import *




if __name__ == '__main__':
    original_image = load_image(
        r"C:/Users/yuval/PycharmProjects/Image and Video Compression/Ex2/Mona-Lisa.bmp")
    # perform_dct(original_image)
    # reconstructed_image = until_quantization_test(original_image, 0.5)
    # imageio.imsave("reconstructed_image_quantization.jpg", reconstructed_image)