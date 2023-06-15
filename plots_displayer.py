from fractions import Fraction
from typing import List

from PIL import ImageDraw
from imageio.v2 import imsave
from matplotlib import pyplot as plt
import os

from JPEG_compressor import *
from quantizer import *


def plot_quantize(delta: float) -> None:
    """
    Plot the quantization process for a given delta value.

    Args:
        delta (float): The quantization step size.
    """

    x = np.linspace(-1, 1, 100)
    restored_x = inverse_quantization(quantize(x, delta), delta)
    plt.plot(x, restored_x)
    plt.xlabel('x: Quantization Values')
    plt.ylabel('Inverse-Q(Q(x))')
    plt.title(f'Quantization with Delta = {delta}')
    plt.savefig('quantization_steps_plot.png', format='jpeg')
    plt.clf()


def plot_psnr_vs_compression_rate_reference(original_image_path: str) -> None:
    """
    Plot the PSNR (Peak Signal-to-Noise Ratio) against compression rate for a reference image.

    Args:
        original_image_path (str): The file path of the original image.
    """

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


def plot_psnr_vs_delta(matrix: np.ndarray, block_sizes: List[int]) -> None:
    """
    Plot the PSNR (Peak Signal-to-Noise Ratio) against the quantization step size for different block sizes.

    Args:
        matrix (np.ndarray): The input matrix/image.
        block_sizes (List[int]): List of block sizes to consider.
    """

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


def compare_distorted_areas() -> None:
    """
    Compare distorted areas of an image.
    """

    image = load_image("Mona-Lisa.bmp")

    areas = [
        np.array([[100, 100], [200, 200]]),
        np.array([[300, 250], [400, 350]]),
        np.array([[350, 50], [450, 150]])
    ]

    range_of_deltas = range(1, 6)

    for i, area in enumerate(areas):
        distortions = []
        for r in range_of_deltas:
            delta = 1 / r
            encoded_im = encode_until_quantization(image, delta, 8)
            decoded_im = decode_from_quantization(encoded_im, delta)
            imsave(f"reconstructed image{i + 1}.png", decoded_im)
            reconstructed_area = decoded_im[area[0][1]:area[1][1], area[0][0]:area[1][0]]
            distortions.append(reconstructed_area)
        area_sequence = np.hstack(distortions)
        imsave(f"area{i + 1}.png", area_sequence)

    # Draw the distorted areas on the image
    marked_image = Image.open(f"reconstructed image2.png")
    draw = ImageDraw.Draw(marked_image)
    for area in areas:
        x_start, y_start = area[0]
        x_end, y_end = area[1]
        draw.rectangle(((x_start, y_start), (x_end, y_end)), outline='yellow')
    marked_image.save(f"marked_areas.png")


def plot_best_lengths(image: np.ndarray, block_sizes: List[int], lens: List[float], psnrs: List[float]) -> plt:
    """
    Plot the PSNR (Peak Signal-to-Noise Ratio) as a function of compression rate.

    Args:
        image (np.ndarray): The input image.
        block_sizes (List[int]): List of block sizes.
        lens (List[float]): List of lengths.
        psnrs (List[float]): List of PSNR values.

    Returns:
        plt: The matplotlib.pyplot object.
    """

    image_size_in_bits = image.size * image.itemsize * 8
    ratios = lens / image_size_in_bits
    for i, n in enumerate(block_sizes):
        plt.plot(ratios[i], psnrs[i], label=f'N={n}')
    plt.xlabel('Compression Ratio')
    plt.ylabel('PSNR')
    plt.title('PSNR as a Function of Compression Rate')
    plt.legend()
    return plt


def get_jpeg_data(image: np.ndarray) -> Tuple[List[float], List[float]]:
    """
    Get the rate and PSNR data for JPEG compression.

    Args:
        image (np.ndarray): The input image.

    Returns:
        Tuple[List[float], List[float]]: The rates and PSNR values.
    """

    qualities = range(1, 31)
    rates, psnrs = zip(*[get_rate_and_psnr_jpeg(image, quality) for quality in qualities])
    return list(rates), list(psnrs)


def get_params_table(block_sizes: List[int], deltas: List[float], params: List[List[float]]) -> pd.DataFrame:
    """
    Generate a pandas DataFrame for the parameters.

    Args:
        block_sizes (List[int]): List of block sizes.
        deltas (List[float]): List of deltas.
        params (List[List[float]]): List of parameter values.

    Returns:
        pd.DataFrame: The DataFrame containing the parameters.
    """
    df = pd.DataFrame(params, index=block_sizes, columns=deltas)
    return df


def plot_psnr_vs_compression_rate(image: np.ndarray) -> None:
    """
    Plot the PSNR (Peak Signal-to-Noise Ratio) against compression rate for different configurations.

    Args:
        image (np.ndarray): The input image.
    """

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


def display_best_parameters() -> None:
    """
    Display the best parameters in a tabular format.
    """

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
    # block_sizes = [8, 16]
    # plot_quantize(0.3)
    # plot_psnr_vs_compression_rate_reference(original_image_path)
    # plot_psnr_vs_delta(original_image, block_sizes)
    # compare_distorted_areas()
    # plot_psnr_vs_compression_rate(original_image)
    # display_best_parameters()


