import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import os

from quantizer import *


def plot_quantize(delta):
    """
    Save a plot of x : inv-Q(Q(x)), that shows the quantization steps function.
    :return:
    """
    x = np.linspace(-1, 1, 100)
    restored_x = inverse_quantization(quantize(x, delta), delta)
    plt.plot(x, restored_x)
    plt.xlabel('x: Quantization Values')
    plt.ylabel('Inverse-Q(Q(x))')
    plt.title(f'Quantization with Delta = {delta}')
    plt.savefig('quantization_steps_plot.png', format='jpeg')
    plt.clf()


def plot_psnr_vs_compression_rate(original_image_path):
    quality_levels = range(10, 96, 5)
    original_image = Image.open(original_image_path)
    original_size = os.path.getsize(original_image_path)
    original_array = np.array(original_image)

    psnr_values = []
    compression_rates = []

    for quality in quality_levels:
        compressed_path = f"compressed_{quality}.jpg"
        original_image.save(compressed_path, quality=quality)
        compressed_image = Image.open(compressed_path)

        compressed_size = os.path.getsize(compressed_path)
        compressed_array = np.array(compressed_image)
        compression_rate = compressed_size / original_size

        psnr_value = calculate_psnr(original_array, compressed_array, 8)
        psnr_values.append(psnr_value)
        compression_rates.append(compression_rate)
        os.remove(compressed_path)

    # Plot PSNR vs. compression rate
    plt.plot(compression_rates, psnr_values, marker='o')
    plt.xlabel('Compression Rate')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR vs. Compression Rate')
    plt.savefig('psnr vs compression rate reference.png', format='jpeg')
    plt.clf()


def plot_psnr_vs_delta(matrix, block_sizes):
    deltas = np.linspace(0.1, 1, 20)

    for block_size in block_sizes:
        psnr_values = []
        for delta in deltas:
            quantized_image = \
                until_quantization_encoder(matrix, delta, block_size)
            reconstructed_image = \
                until_quantization_decoder(quantized_image, delta)
            psnr = calculate_psnr(matrix, reconstructed_image)
            psnr_values.append(psnr)
        plt.plot(deltas, psnr_values, label=f"N={block_size}")

    # Plot PSNR vs. δ for different block sizes
    plt.plot(deltas, 10 * np.log10(12 / (deltas ** 2)), label="Reference")
    plt.xlabel('δ (Quantization Step Size)')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR vs. Quantization Step Size')
    plt.legend()
    plt.savefig('delta_psnr_plot.jpg', format='jpeg')
    plt.show()


if __name__ == '__main__':
    original_image_path = r"Mona-Lisa.bmp"
    original_image = load_image(original_image_path)
    # plot_quantize(0.3)
    # plot_psnr_vs_compression_rate(original_image_path)
    block_sizes = [8, 16]
    plot_psnr_vs_delta(original_image, block_sizes)
    # until_dct_test(original_image, 8)
