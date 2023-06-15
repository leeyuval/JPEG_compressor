from dct import *


def apply_dpcm_encoding(encoded_blocks: np.ndarray, delta: float) -> np.ndarray:
    """
    Apply Differential Pulse Code Modulation (DPCM) encoding to the encoded blocks.

    Args:
        encoded_blocks: Encoded blocks as a numpy array.
        delta: Delta value for quantization.

    Returns:
        Encoded blocks after DPCM encoding as a numpy array.
    """
    quantize = lambda x: np.round(x / delta).astype(int) * delta
    m, n = encoded_blocks.shape[: 2]
    y_matrix = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            if i == 0 and j == 0:
                y_matrix[0, 0] = quantize(encoded_blocks[i, j, 0, 0])
            elif i == 0:
                encoded_blocks[i, j, 0, 0] -= y_matrix[i, j - 1]
                y_matrix[i][j] = y_matrix[i, j - 1] + quantize(encoded_blocks[i, j, 0, 0])
            elif j == 0:
                encoded_blocks[i, j, 0, 0] -= y_matrix[i - 1, j]
                y_matrix[i][j] = y_matrix[i - 1][j] + quantize(encoded_blocks[i, j, 0, 0])
            else:
                encoded_blocks[i, j, 0, 0] -= (y_matrix[i - 1, j] + y_matrix[i, j - 1]) / 2
                y_matrix[i][j] = (y_matrix[i - 1, j] + y_matrix[i, j - 1]) / 2 + quantize(encoded_blocks[i, j, 0, 0])
    return encoded_blocks


def apply_dpcm_decoding(blocks: np.ndarray) -> np.ndarray:
    """
    Apply Differential Pulse Code Modulation (DPCM) decoding to the blocks.

    Args:
        blocks: Encoded blocks as a numpy array.

    Returns:
        Decoded blocks after DPCM decoding as a numpy array.
    """
    m, n = blocks.shape[: 2]
    for i in range(m):
        for j in range(n):
            if i == 0 and j == 0:
                pass
            elif i == 0:
                blocks[i, j, 0, 0] += blocks[i, j - 1, 0, 0]
            elif j == 0:
                blocks[i, j, 0, 0] += blocks[i - 1, j, 0, 0]
            else:
                blocks[i, j, 0, 0] += (blocks[i - 1, j, 0, 0] + blocks[i, j - 1, 0, 0]) / 2
    return blocks


def encode_until_dpcm(image: np.ndarray, delta: float, block_size: int) -> np.ndarray:
    """
    Encode an image until DPCM is applied.

    Args:
        image: Input image as a numpy array.
        delta: Delta value for quantization.
        block_size: Size of the blocks to be used.

    Returns:
        Encoded blocks as a numpy array.
    """
    encoded_blocks = encode_until_dct(image, block_size)
    return apply_dpcm_encoding(encoded_blocks, delta)


def decode_from_dpcm(blocks: np.ndarray, resolution: int) -> np.ndarray:
    """Decode the image from DPCM encoded blocks.

    Args:
        blocks: DPCM encoded blocks.
        resolution: Resolution parameter for DCT decoding.

    Returns:
        Decoded image.

    """
    blocks = apply_dpcm_decoding(blocks)
    image = decode_from_dct(blocks, resolution)
    return image
