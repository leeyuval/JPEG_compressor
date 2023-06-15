from dpcm import *


def quantize(coefficients: np.ndarray, delta: float) -> np.ndarray:
    """
    Quantizes the input coefficients using the specified delta value.

    Args:
        coefficients: The input coefficients to be quantized.
        delta: The quantization step size.

    Returns:
        The quantized coefficients.

    """
    return np.round(coefficients / delta).astype(int)


def inverse_quantization(quantized_coefficients: np.ndarray, delta: float) -> np.ndarray:
    """
    Performs inverse quantization on the input quantized coefficients using the specified delta value.

    Args:
        quantized_coefficients: The quantized coefficients to be inverse quantized.
        delta: The quantization step size.

    Returns:
        The inverse quantized coefficients.

    """
    return quantized_coefficients * delta


def encode_until_quantization(matrix: np.ndarray, delta: float, block_size: int, dpcm: bool = False) -> np.ndarray:
    """
    Encodes the input matrix until quantization using the specified parameters.

    Args:
        matrix: The input matrix to be encoded.
        delta: The quantization step size.
        block_size: The size of the blocks for encoding.
        dpcm: Whether to apply DPCM encoding (default: False).

    Returns:
        The quantized encoded blocks.

    """
    encoded_blocks = encode_until_dct(matrix, block_size)
    if dpcm:
        encoded_blocks = apply_dpcm_encoding(encoded_blocks, delta)
    return quantize(encoded_blocks, delta)


def decode_from_quantization(quantized_matrix: np.ndarray, delta: float, dpcm: bool = False, resolution: int = 8) -> np.ndarray:
    """
    Decodes the input quantized matrix using the specified parameters.

    Args:
        quantized_matrix: The quantized matrix to be decoded.
        delta: The quantization step size.
        dpcm: Whether to apply DPCM decoding (default: False).
        resolution: The resolution for inverse DCT (default: 8).

    Returns:
        The decoded matrix.

    """
    blocks = inverse_quantization(quantized_matrix, delta)
    if dpcm:
        blocks = apply_dpcm_decoding(blocks)
    return decode_from_dct(blocks, resolution)
