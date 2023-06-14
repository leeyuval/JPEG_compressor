from fractions import Fraction

from matplotlib import pyplot as plt
import os

from encoder import *
from quantizer import *


def plot_quantize(delta):
    x = np.linspace(-1, 1, 100)
    restored_x = inverse_quantization(quantize(x, delta), delta)
    plt.plot(x, restored_x)
    plt.xlabel('x: Quantization Values')
    plt.ylabel('Inverse-Q(Q(x))')
    plt.title(f'Quantization with Delta = {delta}')
    plt.savefig('quantization_steps_plot.png', format='jpeg')
    plt.clf()


def plot_psnr_vs_compression_rate_reference(original_image_path):
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
                encode_until_quantization(matrix, delta, block_size)
            reconstructed_image = \
                decode_from_quantization(quantized_image, delta)
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


def plot_best_lengths(image, block_sizes, lens, psnrs):
    image_size_in_bits = image.size * image.itemsize * 8
    ratios = lens / image_size_in_bits
    for i, n in enumerate(block_sizes):
        plt.plot(ratios[i], psnrs[i], label=f'N={n}')
    plt.xlabel('Compression Ratio')
    plt.ylabel('PSNR')
    plt.title('PSNR as a Function of Compression Rate')
    plt.legend()
    return plt


def get_jpeg_data(image: np.ndarray):
    qualities = range(1, 31)
    rates, psnrs = zip(*[get_rate_and_psnr_jpeg(image, quality) for quality in qualities])
    return list(rates), list(psnrs)


def get_params_table(block_sizes, deltas, params):
    df = pd.DataFrame(params, index=block_sizes, columns=deltas)
    return df


def plot_psnr_vs_compression_rate(image):
    deltas = 1 / np.arange(1, 11)
    block_sizes = [8, 16]
    jpeg_rates, jpeg_psnrs = get_jpeg_data(image)

    versions = [NORMAL_VERSION, IMPROVED_VERSION, IMPROVED_VERSION]
    dpcm_flags = [False, False, True]
    filenames = ["4_3_1_4_psnr_ratio_plot.png", "4_3_2_4_psnr_ratio_plot.png", "5_psnr_ratio_plot.png"]
    table_filenames = ["4_3_1_table.csv", "4_3_2_table.csv", "5_table.csv"]

    for version, filename, table_filename, dpcm in zip(versions, filenames, table_filenames, dpcm_flags):
        lens, psnrs, params = find_best_lengths(image, block_sizes, deltas, version=version, dpcm=dpcm)
        plot_best_lengths(image, block_sizes, lens, psnrs)
        plt.plot(jpeg_rates, jpeg_psnrs, label='JPEG')
        plt.legend()
        plt.savefig(f"Plots/{filename}")
        df = get_params_table(block_sizes, deltas, params)
        df.to_csv(f"Plots/{table_filename}")
        plt.clf()

def display_best_parameters():
    table_filenames = ["Plots/4_3_1_table.csv", "Plots/4_3_2_table.csv", "Plots/5_table.csv"]
    for table_filename in table_filenames:
        data = pd.read_csv(table_filename)
        columns = list(data.columns)
        columns[1:] = columns[:0:-1]
        data = data[columns]
        headers = list(data.columns)
        headers[0] = "Block size / Delta(δ)"

        for i in range(1, len(headers)):
            fraction_value = Fraction(headers[i]).limit_denominator()
            headers[i] = f"{fraction_value.numerator}/{fraction_value.denominator}"

        table = data.values.tolist()

        print(tabulate(table, headers, tablefmt="fancy_grid", stralign='center'), '\n')


if __name__ == '__main__':
    original_image_path = r"Mona-Lisa.bmp"
    original_image = load_image(original_image_path)
    display_best_parameters()
    # block_sizes = [8, 16]
    # plot_quantize(0.3)
    # plot_psnr_vs_compression_rate_reference(original_image_path)
    # plot_psnr_vs_delta(original_image, block_sizes)
    # plot_psnr_vs_compression_rate(original_image)


