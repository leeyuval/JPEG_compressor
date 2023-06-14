import imageio.v2 as imageio
import numpy as np


def load_image(bmp_file_path):
    image = imageio.imread(bmp_file_path)
    return image


def save_image(matrix, bmp_file_path):
    imageio.imsave(bmp_file_path, matrix)


def bind_matrix(matrix, d):
    if d <= 0 or matrix.shape[0] % d != 0 or matrix.shape[1] % d != 0:
        raise ValueError("The matrix size is not divisible by d")

    num_rows = matrix.shape[0] // d
    num_cols = matrix.shape[1] // d
    block_matrix = matrix.reshape((num_rows, d, num_cols, d)).swapaxes(1, 2)

    return block_matrix


def unbind_matrix(blocks):
    num_rows, num_cols, d, _ = blocks.shape
    matrix = blocks.swapaxes(1, 2).reshape(num_rows * d, num_cols * d)

    return matrix


def scale_matrix(matrix, k=8):
    k = np.iinfo(matrix.dtype).bits
    matrix = matrix.astype(float)
    matrix = matrix / (2 ** k) - 0.5
    return matrix


def descale_matrix(matrix, k):
    matrix[matrix < -0.5] = -0.5
    matrix = (matrix + 0.5) * (2 ** k)
    matrix = np.rint(matrix).astype(np.uint8)
    return matrix


def zigzag_order(matrix):
    N, M = matrix.shape
    if N != M:
        raise ValueError("Input block is not a square matrix.")

    matrix = np.fliplr(matrix)
    zigzag_vector = np.concatenate([matrix.diagonal(i)[::2 * (i % 2) - 1] for i in range(N - 1, -N, -1)])
    return zigzag_vector


def reverse_zigzag_order(vector: np.ndarray):
    N = int(np.sqrt(len(vector)))
    matrix = np.zeros((N, N), dtype=vector.dtype)

    zigzag_index = 0

    for i in range(N - 1, -N, -1):
        diagonal_length = N - abs(i)
        vals = vector[zigzag_index:zigzag_index + diagonal_length]
        zigzag_index += diagonal_length

        start = 0 if i >= 0 else abs(i)
        rng = np.arange(start, start + diagonal_length)
        ind = rng + i + rng * N
        matrix.flat[ind] = vals[::2 * (i % 2) - 1]

    matrix = np.fliplr(matrix)
    return matrix


def calculate_psnr(X, Y):
    """
    Calculates the peak row_block-to-noise ratio (PSNR) between two images.

    Args:
      x: The first image.
      y: The second image.

    :param Y:
    :param X:

    Returns:
      The PSNR in dB.
    """
    if X.shape != Y.shape:
        raise TypeError("Images must have the same shape")

    k = np.iinfo(X.dtype).bits
    A = 2 ** k - 1

    X = X.astype(float)
    Y = Y.astype(float)

    mse = np.mean((X - Y) ** 2)
    psnr_db = 20 * np.log10(A / np.sqrt(mse))
    return psnr_db
