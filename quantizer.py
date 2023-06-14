from dpcm import *

def quantize(coefficients, delta):
    return np.round(coefficients / delta).astype(int)


def inverse_quantization(quantized_coefficients, delta):
    return quantized_coefficients * delta


def encode_until_quantization(matrix, delta, block_size, dpcm=False):
    encoded_blocks = encode_until_dct(matrix, block_size)
    if dpcm:
        encoded_blocks = apply_dpcm_encoding(encoded_blocks, delta)
    return quantize(encoded_blocks, delta)


def decode_from_quantization(quantized_matrix, delta, dpcm=False, resolution: int = 8):
    blocks = inverse_quantization(quantized_matrix, delta)
    if dpcm:
        blocks = apply_dpcm_decoding(blocks)
    return decode_from_dct(blocks, resolution)
