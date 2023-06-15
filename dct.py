from image_operations import *


def dct_1d(vector: np.ndarray) -> np.ndarray:
    """
    Perform 1D Discrete Cosine Transform (DCT) on a signal.

    Args:
        vector: Input signal as a 1D numpy array.

    Returns:
        DCT-transformed signal as a 1D numpy array.
    """
    N = vector.size
    alpha = np.full(shape=N, fill_value=np.sqrt(2 / N))
    alpha[0] = 1 / np.sqrt(N)

    cos_values = np.cos((np.pi * (2 * np.arange(N)[:, np.newaxis] + 1) *
                         np.arange(N)) / (2 * N))

    dct_vector = (vector @ cos_values) * alpha

    return dct_vector


def dct_2d(matrix: np.ndarray) -> np.ndarray:
    """
    Perform 2D Discrete Cosine Transform (DCT) over a 2D matrix.

    Args:
        matrix: Input matrix (image) as a 2D numpy array.

    Returns:
        DCT-transformed matrix as a 2D numpy array.
    """
    dct_on_rows = np.apply_along_axis(dct_1d, axis=0, arr=matrix)
    return np.apply_along_axis(dct_1d, axis=1, arr=dct_on_rows)


def idct_1d(coefficients: np.ndarray) -> np.ndarray:
    """
    Perform 1D Inverse Discrete Cosine Transform (IDCT) on coefficients.

    Args:
        coefficients: Input coefficients as a 1D numpy array.

    Returns:
        IDCT-transformed signal as a 1D numpy array.
    """
    N = coefficients.size
    alpha = np.full(shape=N, fill_value=np.sqrt(2 / N))
    alpha[0] = 1 / np.sqrt(N)
    cos_values = np.cos((np.pi * (2 * np.arange(N)[:, np.newaxis] + 1) *
                         np.arange(N)) / (2 * N))
    dct_vec = cos_values @ (coefficients * alpha)
    return dct_vec


def idct_2d(coefficients: np.ndarray) -> np.ndarray:
    """
    Perform 2D Inverse Discrete Cosine Transform (IDCT) on coefficients.

    Args:
        coefficients: Input coefficients as a 2D numpy array.

    Returns:
        IDCT-transformed matrix as a 2D numpy array.
    """
    idct_on_cols = np.apply_along_axis(idct_1d, axis=1, arr=coefficients)
    return np.apply_along_axis(idct_1d, axis=0, arr=idct_on_cols)


def apply_dct_encoding(blocks: np.ndarray) -> np.ndarray:
    """
    Apply DCT encoding to a set of blocks.

    Args:
        blocks: Input blocks as a 4D numpy array.

    Returns:
        Encoded blocks as a 4D numpy array.
    """
    original_shape = blocks.shape
    m, n, d, _ = original_shape
    blocks_vector = blocks.reshape((-1, d, d))
    blocks_vector = np.array([dct_2d(block) for block in blocks_vector])
    dct_blocks_image = blocks_vector.reshape(original_shape)
    return dct_blocks_image


def apply_dct_decoding(blocks: np.ndarray) -> np.ndarray:
    """
    Apply DCT decoding to a set of blocks.

    Args:
        blocks: Input blocks as a 4D numpy array.

    Returns:
        Decoded blocks as a 4D numpy array.
    """
    original_shape = blocks.shape
    m, n, d, _ = original_shape
    blocks_vector = blocks.reshape((-1, d, d))
    blocks_vector = np.array([idct_2d(block) for block in blocks_vector])
    blocks_image = blocks_vector.reshape(original_shape)
    return blocks_image


def encode_until_dct(image: np.ndarray, block_size: int) -> np.ndarray:
    """
    Encode an image until DCT is applied.

    Args:
        image: Input image as a numpy array.
        block_size: Size of the blocks to be used.

    Returns:
        Encoded blocks as a numpy array.
    """
    scale = scale_matrix(image)
    blocks = bind_matrix(scale, block_size)
    return apply_dct_encoding(blocks)


def decode_from_dct(blocks: np.ndarray, resolution: int = 8) -> np.ndarray:
    """
    Decode blocks from DCT back to the original image.

    Args:
        blocks: Encoded blocks as a numpy array.
        resolution: Resolution parameter for descaling.

    Returns:
        Decoded image as a numpy array.
    """
    blocks_image = apply_dct_decoding(blocks)
    scaled_image = unbind_matrix(blocks_image)
    image = descale_matrix(scaled_image, k=resolution)
    return image
