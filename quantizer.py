from dct import *


def quantize(coefficients, delta):
    """
    Quantizes the DCT coefficients.

    Args:
        coefficients (ndarray): DCT coefficients array.
        delta (float): Quantization step size.

    Returns:
        ndarray: Quantized coefficients array.
    """
    return np.round(coefficients / delta).astype(int)


def inverse_quantization(quantized_coefficients, delta):
    """
    Converts quantized coefficients back to DCT coefficients.

    Args:
        quantized_coefficients (ndarray): Quantized coefficients array.
        delta (float): Quantization step size.

    Returns:
        ndarray: DCT coefficients array.
    """
    return quantized_coefficients * delta


def until_quantization_encoder(matrix, delta, block_size):
    after_dct = encode_until_dct(matrix, block_size)
    return quantize(after_dct, delta)


def until_quantization_decoder(quantized_matrix, delta):
    return decode_until_dct(inverse_quantization(quantized_matrix, delta))

