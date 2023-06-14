from io import BytesIO

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import os

from tqdm import tqdm

from encoder import *
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
    deltas = 1 / np.arange(1, 11)

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


def test_entropy_stage(image):
    """
    Verify that the entropy encoding stage is reversible, i.e that the indices given to the encoder
    are returned by the decoder.
    :param image: image to test.
    :return:
    """
    d = 8
    delta = .2
    k = 5
    enc_blocks = until_quantization_encoder(image, delta, d)
    enc_string = encode_image_blocks(enc_blocks, d, k, method=GOLOMB_RICE, version=NORMAL_VERSION)
    dec_blocks = decode_image_blocks(enc_string, d, k, image.shape, method=GOLOMB_RICE, version=NORMAL_VERSION)
    assert (enc_blocks == dec_blocks).all()


def test_length_calculation(image):
    block_size = 8
    delta = 0.2
    k = 5
    enc_blocks = until_quantization_encoder(image, delta, block_size)
    for version in [NORMAL_VERSION, IMPROVED_VERSION]:
        length = calculate_encoding_length(enc_blocks, k, method=EXPONENTIAL_GOLOMB, version=version)
        encoded_image = encode_image_blocks(enc_blocks, block_size, k, method=EXPONENTIAL_GOLOMB, version=version)
    actual_length = len(encoded_image)
    assert actual_length == length


def calculate_best_length(blocks, version=NORMAL_VERSION):
    """
    Return the best length of encoding for this block, and the parameters that achieve it
    :param improved: whether to compress the nln-zero-value or not.
    :param blocks: block to encode
    :return: length, (bool, param) where bool is flag for golomb-rice or exp-golomb.
    """
    ks = list(range(1, 11))
    Ms = list(range(1, 11))
    best_length = np.inf
    best_params = None
    for k in ks:
        length = calculate_encoding_length(blocks, k, method=EXPONENTIAL_GOLOMB, version=version)
        if length < best_length:
            best_length = length
            best_params = f'exponential-golomb: {k}'
    for m in Ms:
        length = calculate_encoding_length(blocks, m, method=GOLOMB_RICE, version=version)
        if length < best_length:
            best_length = length
            best_params = f'golomb-rice: {m}'
    return best_length, best_params


def best_lengths(image, Ns, deltas, version=NORMAL_VERSION):
    """
    Find lengths and psnr for image.
    :param dpcm:
    :param improved:
    :param image:
    :param Ns:
    :param deltas:
    :return:
    """
    len_results = np.zeros(shape=(len(Ns), len(deltas)))
    psnr_results = np.zeros(shape=(len(Ns), len(deltas)))
    params_results = [[None for _ in range(len(deltas))] for __ in range(len(Ns))]
    for i, n in enumerate(Ns):
        for j, delta in enumerate(tqdm(deltas)):
            blocks = until_quantization_encoder(image, delta, n)
            length, params = calculate_best_length(blocks, version=version)
            rec = until_quantization_decoder(blocks, delta)
            psnr_results[i, j] = calculate_psnr(image, rec)
            len_results[i, j] = length
            params_results[i][j] = params
    return len_results, psnr_results, params_results


def plot_best_lengths(image, Ns, lens, psnrs):
    """
    return a plot of PSNR-ratio.
    :param psnrs:
    :param lens:
    :param Ns:
    :param image:
    :return:
    """
    image_size_in_bits = image.size * image.itemsize * 8
    cratio = lens / image_size_in_bits
    for i, n in enumerate(Ns):
        plt.plot(cratio[i], psnrs[i], label=f'N={n}')
        plt.xlabel('Ratio')
        plt.ylabel('PSNR')
        plt.title('PSNR as a function of Compression Rate')
    return plt

def get_rate_and_psnr_jpeg(im_array, quality):
    """
    return PSNR and compression rate when compressing with a standard JPEG library.
    :param im_array: image array to compress
    :param quality: jpeg compression quality parameter
    :return: rate, PSNR
    """
    before_size_bytes = im_array.size * im_array.itemsize

    # save image to buffer as jpeg.
    buffer = BytesIO()
    image = Image.fromarray(im_array)
    image.save(buffer, 'JPEG', quality=quality)
    buffer.seek(0)

    # compute rate R
    after_size_bytes = buffer.getbuffer().nbytes
    R = after_size_bytes / before_size_bytes

    # compute PSNR
    rec_image = np.asarray(Image.open(buffer))
    psnr = calculate_psnr(rec_image, im_array)
    return R, psnr
def get_jpeg_data(image: np.ndarray):
    """
    Get rate-PSNR curve for PIL.Image JPEG Compressor.
    :return:
    """
    qualities = list(range(1, 31))
    rates = []
    psnrs = []
    for quality in qualities:
        rate, psnr = get_rate_and_psnr_jpeg(image, quality)
        rates.append(rate)
        psnrs.append(psnr)
    return rates, psnrs
def get_params_table(Ns, deltas, params):
    """
    Craete a df for params
    :param Ns:
    :param deltas:
    :param params:
    :return:
    """
    df = pd.DataFrame(params)
    df.index = Ns
    df.columns = deltas
    return df

def save_plots_psnr_ratio(image):
    """
    save plots of psnr-compression-rate ratio.
    :param image:
    :return:
    """
    deltas = 1 / np.arange(1, 11)
    Ns = [8, 16]
    jpeg_rates, jpeg_psnrs = get_jpeg_data(image)

    lens, psnrs, params = best_lengths(image, Ns, deltas, version=NORMAL_VERSION)
    plot_best_lengths(image, Ns, lens, psnrs)
    plt.plot(jpeg_rates, jpeg_psnrs, label='JPEG')
    plt.legend()
    plt.savefig(r"Plots/4_3_1_4_psnr_ratio_plot.png")
    df = get_params_table(Ns, deltas, params)
    df.to_csv("Plots/4_3_1_table.csv")

    plt.clf()

    lens, psnrs, params = best_lengths(image, Ns, deltas, version=IMPROVED_VERSION)
    plot_best_lengths(image, Ns, lens, psnrs)
    plt.plot(jpeg_rates, jpeg_psnrs, label='JPEG')
    plt.legend()
    plt.savefig(r"Plots/4_3_2_4_psnr_ratio_plot.png")
    df = get_params_table(Ns, deltas, params)
    df.to_csv("Plots/4_3_2_table.csv")

    plt.clf()


if __name__ == '__main__':
    original_image_path = r"Mona-Lisa.bmp"
    original_image = load_image(original_image_path)
    save_plots_psnr_ratio(original_image)
    # test_length_calculation(original_image)
    # test_entropy_stage(original_image)
    # plot_quantize(0.3)
    # plot_psnr_vs_compression_rate(original_image_path)
    # block_sizes = [8, 16]
    # plot_psnr_vs_delta(original_image, block_sizes)
    # until_dct_test(original_image, 8)
