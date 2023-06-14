from image_operations import *


def dct_1d(vector):
    """Perform 1D Discrete Cosine Transform (DCT) on a signal."""
    N = vector.size
    alpha = np.full(shape=N, fill_value=np.sqrt(2 / N))
    alpha[0] = 1 / np.sqrt(N)

    cos_values = np.cos((np.pi * (2 * np.arange(N)[:, np.newaxis] + 1) *
                         np.arange(N)) / (2 * N))

    dct_vector = (vector @ cos_values) * alpha

    return dct_vector


def dct_2d(matrix):
    """Perform 2D Discrete Cosine Transform (DCT) over a 2D matrix."""
    dct_on_rows = np.apply_along_axis(dct_1d, axis=0, arr=matrix)
    return np.apply_along_axis(dct_1d, axis=1, arr=dct_on_rows)


def idct_1d(coefficients):
    """Perform 1D Inverse Discrete Cosine Transform (IDCT) on coefficients."""
    N = coefficients.size
    alpha = np.full(shape=N, fill_value=np.sqrt(2 / N))
    alpha[0] = 1 / np.sqrt(N)
    cos_values = np.cos((np.pi * (2 * np.arange(N)[:, np.newaxis] + 1) *
                         np.arange(N)) / (2 * N))
    dct_vec = cos_values @ (coefficients * alpha)
    return dct_vec


def idct_2d(coefficients):
    """Perform 2D Inverse Discrete Cosine Transform (IDCT) on coefficients."""
    idct_on_cols = np.apply_along_axis(idct_1d, axis=1, arr=coefficients)
    return np.apply_along_axis(idct_1d, axis=0, arr=idct_on_cols)


def apply_dct_encoding(blocks: np.ndarray):
    original_shape = blocks.shape
    m, n, d, _ = original_shape
    blocks_vector = blocks.reshape((-1, d, d))
    blocks_vector = np.array([dct_2d(block) for block in blocks_vector])
    dct_blocks_image = blocks_vector.reshape(original_shape)
    return dct_blocks_image


def apply_dct_decoding(blocks: np.ndarray):
    original_shape = blocks.shape
    m, n, d, _ = original_shape
    blocks_vector = blocks.reshape((-1, d, d))
    blocks_vector = np.array([idct_2d(block) for block in blocks_vector])
    blocks_image = blocks_vector.reshape(original_shape)
    return blocks_image


def encode_until_dct(image: np.ndarray, block_size: int):
    scale = scale_matrix(image)
    blocks = bind_matrix(scale, block_size)
    return apply_dct_encoding(blocks)


def decode_from_dct(blocks: np.ndarray, resolution: int = 8):
    blocks_image = apply_dct_decoding(blocks)
    scaled_image = unbind_matrix(blocks_image)
    image = descale_matrix(scaled_image, k=resolution)
    return image
