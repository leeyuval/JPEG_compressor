import imageio.v2 as imageio
import numpy as np
from scipy.fft import dct, idct


def perform_dct(image):
    """Performs 2D DCT on an image.

  Args:
    image: A NumPy array of shape (height, width, channels).

  Returns:
    A NumPy array of shape (height, width, channels) containing the DCT coefficients.
  """

    return dct(image)


def quantize(dct_coefficients, quantization_table):
    """Quantizes a set of DCT coefficients using a quantization table.

  Args:
    dct_coefficients: A NumPy array of shape (height, width, channels) containing the DCT coefficients.
    quantization_table: A NumPy array of shape (height, width, channels) containing the quantization factors.

  Returns:
    A NumPy array of shape (height, width, channels) containing the quantized DCT coefficients.
  """

    quantized_coefficients = np.round(dct_coefficients / quantization_table)
    return quantized_coefficients


def zigzag(coefficients):
    """Rearranges a set of coefficients in zigzag order.

  Args:
    coefficients: A NumPy array of shape (height, width, channels).

  Returns:
    A NumPy array of shape (height * width * channels,) containing the coefficients in zigzag order.
  """

    zigzag_order = np.array([
        0, 1, 8, 16, 9, 2, 10, 17,
        18, 11, 3, 12, 19, 24, 25, 13,
        4, 14, 20, 26, 27, 15, 5, 21,
        28, 29, 6, 22, 30, 31, 7, 23
    ])
    zigzag_coefficients = np.zeros_like(coefficients)
    for i in range(coefficients.shape[0]):
        for j in range(coefficients.shape[1]):
            for k in range(coefficients.shape[2]):
                zigzag_coefficients[
                    zigzag_order[i * coefficients.shape[1] * coefficients.shape[2] + j * coefficients.shape[2] + k]] = \
                    coefficients[i, j, k]
    return zigzag_coefficients


def entropy_encode(coefficients):
    """Encodes a set of coefficients using entropy coding.

  Args:
    coefficients: A NumPy array of shape (height * width * channels,).

  Returns:
    A string containing the encoded coefficients.
  """

    entropy_code = ""
    for coefficient in coefficients:
        entropy_code += chr(int(coefficient))
    return entropy_code


def image_compression(image, quantization_table):
    """Compresses an image using DCT-based compression.

  Args:
    image: A NumPy array of shape (height, width, channels).
    quantization_table: A NumPy array of shape (height, width, channels) containing the quantization factors.

  Returns:
    A string containing the compressed image.
  """

    dct_coefficients = perform_dct(image)
    quantized_coefficients = quantize(dct_coefficients, quantization_table)
    zigzag_coefficients = zigzag(quantized_coefficients)
    entropy_code = entropy_encode(zigzag_coefficients)
    return entropy_code


def decode_image(entropy_code, quantization_table):
    """Decodes an image from a string of entropy-encoded coefficients.

  Args:
    entropy_code: A string containing the entropy-encoded coefficients.
    quantization_table: A NumPy array of shape (height, width, channels) containing the quantization factors.

  Returns:
    A NumPy array of shape (height, width, channels) containing the reconstructed image.
  """

    zigzag_coefficients = []
    for char in entropy_code:
        zigzag_coefficients.append(int(char))

    coefficients = zigzag_coefficients[::-1]
    for i in range(len(coefficients)):
        coefficients[i] = float(coefficients[i])

    dequantized_coefficients = coefficients * quantization_table
    inverse_zigzag_coefficients = np.zeros_like(coefficients)
    for i in range(coefficients.shape[0]):
        for j in range(coefficients.shape[1]):
            for k in range(coefficients.shape[2]):
                inverse_zigzag_coefficients[
                    i * coefficients.shape[1] * coefficients.shape[2] + j * coefficients.shape[2] + k] = coefficients[
                    zigzag_coefficients[i * coefficients.shape[1] * coefficients.shape[2] + j * coefficients.shape[2] + k]]

    inverse_dct_coefficients = idct(inverse_zigzag_coefficients, axis=None, norm='ortho')

    return inverse_dct_coefficients


def main():
    # Load the original image.
    image = imageio.imread(r"C:/Users/yuval/PycharmProjects/Image and Video Compression/Ex2/Mona-Lisa.bmp")

    # # Get the shape of the image.
    # height, width, channels = image.shape

    # Create a quantization table.
    quantization_table = np.array([
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 55, 69, 109, 104, 77,
        24, 35, 55, 64, 82, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99
    ])

    # Perform DCT on the image.
    dct_coefficients = dct(image)

    # Quantize the DCT coefficients.
    quantized_coefficients = np.round(dct_coefficients / quantization_table)

    # Rearrange the quantized coefficients in zigzag order.
    zigzag_coefficients = zigzag(quantized_coefficients)

    # Encode the zigzag coefficients using entropy coding.
    entropy_code = entropy_encode(zigzag_coefficients)

    # Decode the entropy-encoded coefficients.
    dequantized_coefficients = decode_image(entropy_code, quantization_table)

    # Reconstruct the original image from the dequantized coefficients.
    inverse_dct_coefficients = idct(dequantized_coefficients, axis=None, norm='ortho')

    # Compare the original image to the reconstructed image.
    imageio.imshow("Original Image", image)
    imageio.imshow("Reconstructed Image", inverse_dct_coefficients)
    imageio.waitKey(0)


if __name__ == "__main__":
    main()
