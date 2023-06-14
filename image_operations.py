import imageio.v2 as imageio
import numpy as np


def load_image(bmp_file_path):
    image = imageio.imread(bmp_file_path)
    return image


def save_image(matrix, bmp_file_path):
    imageio.imsave(bmp_file_path, matrix)


def bind_matrix(matrix, d):
    """
    Splits a matrix into sub-matrices of size d x d and stores them
    in a block structure.

    Parameters: matrix (numpy.ndarray): A 2-dimensional numpy ndarray
    representing the matrix to be binned.
    d (int): An integer representing the size of the sub-matrices.

    Returns: numpy.ndarray: A 4-dimensional numpy ndarray representing the
    block-structured matrix.

    Raises:
        ValueError: If the matrix size is not divisible by d.

    """

    if d <= 0 or matrix.shape[0] % d != 0 or matrix.shape[1] % d != 0:
        raise ValueError("The matrix size is not divisible by d")

    num_rows = matrix.shape[0] // d
    num_cols = matrix.shape[1] // d
    block_matrix = matrix.reshape(num_rows, num_cols, d, d)

    return block_matrix


def unbind_matrix(blocks):
    """
    Reconstructs a matrix from a block structure.

    Parameters: blocks (numpy.ndarray): A 4-dimensional numpy ndarray
    representing the block-structured matrix.

    Returns: numpy.ndarray: A 2-dimensional numpy ndarray representing the
    reconstructed matrix.

    """

    num_rows, num_cols, d, _ = blocks.shape
    matrix = blocks.reshape(num_rows * d, num_cols * d)

    return matrix


def scale_matrix(matrix, k):
    """
    Scales each element of a matrix from the range [0, 2^K - 1] to
    the range [-1/2, 2^(K-1) - 1/2^K], where K is the resolution in bits.

    Parameters:
        matrix (numpy.ndarray): A 2-dimensional numpy ndarray representing the matrix
            to be scaled.
        k (int): An integer representing the resolution in bits.

    Returns:
        numpy.ndarray: A 2-dimensional numpy ndarray representing the scaled matrix.

    """
    return (matrix - 2 ** (k - 1)) / 2 ** k


def descale_matrix(matrix, k):
    """
    Descales each element of a matrix from the range [-1/2, 2^(K-1) - 1/2^K] to
    the range [0, 2^K - 1], where K is the resolution in bits.

    Parameters:
        matrix (numpy.ndarray): A 2-dimensional numpy ndarray representing the scaled matrix
            to be descaled.
        k (int): An integer representing the resolution in bits.

    Returns:
        numpy.ndarray: A 2-dimensional numpy ndarray representing the descaled matrix.

    """
    return (matrix * 2 ** k) + 2 ** (k - 1)


def zigzag_order(block):
    """
    Converts an N × N block of reals to a vector in the zigzag order.

    Args:
    matrix: The input matrix.

    Returns:
    The zigzag vector.
    """

    N, M = block.shape
    if N != M:
        raise ValueError("Input block is not a square matrix.")

    vector_length = N * N
    vector = np.zeros(vector_length)

    i = 0
    for d in range(N + N - 1):
        indices = list(range(max(0, d - N + 1), min(d, N - 1) + 1))
        if d % 2 == 0:
            indices.reverse()
        for j in indices:
            vector[i] = block[j, d - j]
            i += 1

    return vector


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
