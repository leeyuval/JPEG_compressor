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


def encode_dct_stage(blocks: np.ndarray):
    """
    Perform the dct stage of encoding.
    :param blocks:
    :return:
    """
    original_shape = blocks.shape
    m, n, d, _ = original_shape
    blocks_vector = blocks.reshape((-1, d, d))
    blocks_vector = np.array([dct_2d(block) for block in blocks_vector])
    dct_blocks_image = blocks_vector.reshape(original_shape)
    return dct_blocks_image


def decode_dct_stage(blocks: np.ndarray):
    """
    get a dct matrix and decode to image
    :param blocks: blocks of dtc matrices
    :return:
    """
    original_shape = blocks.shape
    m, n, d, _ = original_shape
    blocks_vector = blocks.reshape((-1, d, d))
    blocks_vector = np.array([idct_2d(block) for block in blocks_vector])
    blocks_image = blocks_vector.reshape(original_shape)
    return blocks_image


def encode_until_dct(image: np.ndarray, d: int):
    """
    get an image and do the encode pipline till the dct
    :param image: image matrix in uint
    :param d: block size
    :return:
    """
    scale = scale_matrix(image)
    blocks = bind_matrix(scale, d)
    return encode_dct_stage(blocks)


def decode_until_dct(blocks: np.ndarray, resolution: int = 8):
    """
    get a dct matrix and decode to image
    :param blocks: blocks of dtc matrices
    :param resolution: the image resolution
    :return:
    """
    blocks_image = decode_dct_stage(blocks)
    scaled_image = unbind_matrix(blocks_image)
    image = descale_matrix(scaled_image, k=resolution)
    return image

# def until_dct_encoder(matrix, block_size):
#     # Scale the matrix
#     k = np.iinfo(matrix.dtype).bits
#     scaled_matrix = scale_matrix(matrix, k)
#
#     # Perform blocking
#     blocks = bind_matrix(scaled_matrix, block_size)
#
#     # Apply DCT on each block
#     dct_coefficients = dct_2d(blocks)
#
#     return dct_coefficients
#
#
# def until_dct_decoder(coefficients):
#     # Reconstruct the image from the quantized coefficients
#     reconstructed_blocks = idct_2d(coefficients)
#     reconstructed_image = unbind_matrix(reconstructed_blocks)
#
#     # Rescale the reconstructed image
#     # k = np.iinfo(coefficients.dtype).bits
#     original_image = descale_matrix(reconstructed_image, 8)
#
#     return original_image.astype(np.uint8)


def until_dct_test(matrix, block_size):
    # Encode the image
    encoded_coeffs = encode_until_dct(matrix, block_size)

    # Decode the coefficients
    reconstructed_image = decode_until_dct(encoded_coeffs).astype(np.uint8)

    imageio.imsave("Plots/reconstructed_image.jpg", reconstructed_image)

    # Compare the reconstructed image with the original image
    mse = np.mean((matrix - reconstructed_image) ** 2)
    if mse < 0.5:
        print("The reconstructed picture is identical to the original.")
    else:
        print("The reconstructed picture is not identical to the original.")