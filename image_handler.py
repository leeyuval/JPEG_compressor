import numpy as np
import imageio.v2 as imageio

GOLOMB_RICE = 0
EXPONENTIAL_GOLOMB = 1


# Q1 - Loading a gray BMP image #
def load_image(bmp_file_path):
    image = imageio.imread(bmp_file_path)
    return image


def save_image(matrix, bmp_file_path):
    imageio.imsave(bmp_file_path, matrix)


# Q2 - Flip matrix #
def flip_horizontally(matrix):
    flipped_matrix = np.flip(matrix, axis=1)
    return flipped_matrix


def flip_vertically(matrix):
    flipped_matrix = np.flip(matrix, axis=0)
    return flipped_matrix


# Q3 - Negative image #
def negative_image(image):
    # Check valid data type of the image input
    valid_dtypes = [np.uint8, np.uint16, np.uint32]
    if image.dtype not in valid_dtypes:
        raise ValueError(
            "Input image must have data type of either uint 8, 16 or 32")
    # Get the maximum value of the data type used by the matrix
    max_val = np.iinfo(image.dtype).max

    # Subtract each pixel value from the maximum value to get the negative
    # image
    neg_image = max_val - image

    return neg_image


# Q4 - White frame\padding #
def add_white_padding(matrix, d):
    # Validate input
    if d < 0:
        raise ValueError("d must be a non-negative integer")

    # Get the dimensions of the input matrix
    m, n = matrix.shape

    # Create a new matrix of size (m + 2d, n + 2d) filled with white pixels
    padded_matrix = np.full((m + 2 * d, n + 2 * d), 255, dtype=matrix.dtype)

    # Copy the input matrix onto the padded matrix, leaving a border of width d
    padded_matrix[d:m + d, d:n + d] = matrix

    return padded_matrix


# Q5 - Defining a suitable structure #
def create_picture_structure(num_blocks, block_sizes, dtype):
    # Create an empty numpy ndarray with the appropriate shape
    picture = np.zeros(
        (num_blocks[0], num_blocks[1], block_sizes[0], block_sizes[1]),
        dtype=dtype)

    return picture


# Q6 - Binning a matrix #
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
    # Get the dimensions of the input matrix
    m, n = matrix.shape

    # Check if the matrix size is divisible by d
    if d <= 0 or n % d != 0 or m % d != 0:
        raise ValueError("The matrix size is not divisible by d")

    num_blocks = (m // d, n // d)
    block_sizes = (d, d)

    # Create an empty block-structured matrix
    blocks = create_picture_structure(num_blocks, block_sizes, matrix.dtype)

    # Iterate over the blocks and fill them with the sub-matrices
    for i in range(num_blocks[0]):
        for j in range(num_blocks[1]):
            block = matrix[i * d:(i + 1) * d, j * d:(j + 1) * d]
            blocks[i, j] = block

    return blocks


# Q7 - Un-binning a matrix #
def unbind_matrix(blocks):
    """
    Reconstructs a matrix from a block structure.

    Parameters: blocks (numpy.ndarray): A 4-dimensional numpy ndarray
    representing the block-structured matrix.

    Returns: numpy.ndarray: A 2-dimensional numpy ndarray representing the
    reconstructed matrix.

    """
    # Restore the matrix's dimensions
    num_blocks = blocks.shape[:2]
    block_size = blocks.shape[2:]

    m = num_blocks[0] * block_size[0]
    n = num_blocks[1] * block_size[1]

    # Create an empty matrix
    matrix = np.zeros((m, n), dtype=blocks.dtype)

    # Iterate over the blocks and fill the matrix with the sub-matrices
    for i in range(num_blocks[0]):
        for j in range(num_blocks[1]):
            block = blocks[i, j]
            matrix[i * block_size[0]:(i + 1) * block_size[0],
            j * block_size[1]:(j + 1) * block_size[1]] = block

    return matrix


# Q8 - Clear a sub-matrix
def clear_sub_matrix(matrix, d, i, j):
    """
    Clears the block at index (i,j) in the BMP image.

    Parameters:
        matrix (np.ndarray)
        d (int): The size of the blocks.
        i (int): The row index of the block to clear.
        j (int): The column index of the block to clear.

    Returns:
        PIL.Image.Image: A PIL image object with the block cleared.

    """
    # Binning the image
    blocks = bind_matrix(matrix, d)
    # Clearing the sub-matrix
    blocks[i, j] = np.iinfo(matrix.dtype).max
    # Un-binning the image
    cleared_image = unbind_matrix(blocks)
    return cleared_image


# Q9 - Flipping sub matrices
def flip_sub_matrices(matrix, d):
    """
    Splits an input BMP image into sub-images of size d x d, flips each sub-image
    horizontally and vertically, and returns the resulting BMP image with the same
    size as the input image.

    Parameters:
    matrix (np.ndarray)
    d (int): The size of the sub-images.

    Returns:
    numpy.ndarray: A numpy ndarray representing the processed BMP image.
    """
    # Bin the matrix into sub-matrices
    blocks = bind_matrix(matrix, d)

    # Flip each sub-matrix in both directions
    flipped_blocks = np.flip(np.flip(blocks, axis=2), axis=3)

    # Un-bin the matrix and return the resulting image
    flipped_image = unbind_matrix(flipped_blocks)

    return flipped_image


# Q10 - Padding sub matrices #
def pad_sub_matrices(matrix: np.ndarray, sub_matrix_size: int,
                     padding_size: int):
    """
    Pad the sub-matrices with padding of size 'padding_size'.
    :param sub_matrix_size: size of sub-matrix.
    :param padding_size: size of padding.
    :return: matrix with padded sub-matrices.
    """
    if padding_size <= 0:
        raise ValueError("Parameter 'padding_size' must be positive")

    rows, cols = matrix.shape
    num_sub_matrices_rows = rows // sub_matrix_size
    num_sub_matrices_cols = cols // sub_matrix_size

    binary_matrix = bind_matrix(matrix, sub_matrix_size)
    padded_matrix = np.full(
        shape=(rows + (num_sub_matrices_rows - 1) * padding_size, cols +
               (num_sub_matrices_cols - 1) * padding_size),
        fill_value=np.iinfo(matrix.dtype).max, dtype=matrix.dtype)

    for i in range(num_sub_matrices_rows):
        for j in range(num_sub_matrices_cols):
            sub_matrix_row_start = i * (sub_matrix_size + padding_size)
            sub_matrix_row_end = sub_matrix_row_start + sub_matrix_size
            sub_matrix_col_start = j * (sub_matrix_size + padding_size)
            sub_matrix_col_end = sub_matrix_col_start + sub_matrix_size

            padded_matrix[sub_matrix_row_start:sub_matrix_row_end,
                          sub_matrix_col_start:sub_matrix_col_end] = \
                binary_matrix[i, j]

    return padded_matrix


if __name__ == '__main__':
    im = load_image(
        r"C:/Users/yuval/PycharmProjects/Image and Video Compression/Ex1/Mona-Lisa.bmp")
