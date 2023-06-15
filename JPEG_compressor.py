from io import BytesIO
from PIL import Image

import numpy as np
from tqdm import tqdm

from entropy_encoder import *
from quantizer import *

NORMAL_VERSION = 0
IMPROVED_VERSION = 1

#############################################################
######################## Encoder ############################
#############################################################


def encode_image(image, block_size: int, delta: float, param: int, method: int,
                 dpcm: bool = False, version: int = NORMAL_VERSION):
    quantized_blocks = encode_until_quantization(image, delta, block_size, dpcm)
    encoded_string = encode_image(quantized_blocks, block_size, param, method, version)
    return encoded_string


def encode_image_blocks(blocks, block_size: int, param: int, method: int, version: int):
    encoded_string = ""
    m, n = blocks.shape[:2]
    for i in range(m):
        for j in range(n):
            encoded_string += encode_block(blocks[i, j], block_size, param, method, version)
    return encoded_string


def encode_block(block, block_size: int, param: int, method: int, version: int = NORMAL_VERSION):
    zigzag_vector = zigzag_order(block)
    non_zeros_indices = np.nonzero(zigzag_vector)[0]
    furthest_nonzero_index = 0 if len(non_zeros_indices) == 0 else non_zeros_indices[-1]
    num_of_bits = np.ceil(np.log2(block_size ** 2)).astype(int)
    if version == IMPROVED_VERSION:
        encoded_string = encode_symbol("", furthest_nonzero_index, method, False, param)
    else:
        encoded_string = np.binary_repr(furthest_nonzero_index, width=num_of_bits)
    for i in range(furthest_nonzero_index + 1):
        encoded_string = encode_symbol(encoded_string, zigzag_vector[i], method, True, param)
    return encoded_string


def get_rate_and_psnr_jpeg(im_array, quality):
    before_size_bytes = im_array.nbytes
    buffer = BytesIO()
    Image.fromarray(im_array).save(buffer, 'JPEG', quality=quality)
    after_size_bytes = buffer.tell()
    R = after_size_bytes / before_size_bytes
    rec_image = np.asarray(Image.open(buffer))
    psnr = calculate_psnr(rec_image, im_array)
    return R, psnr


#############################################################
######################## Decoder ############################
#############################################################


def decode_image(encoded_string: str, block_size: int, delta: float, param: int,
                 shape: tuple, method: int, dpcm: bool = False, version: bool = NORMAL_VERSION):
    blocks = decode_image_blocks(encoded_string, block_size, param, shape, method, version)
    return decode_from_quantization(blocks, delta, dpcm)


def decode_image_blocks(encoded_string: str, block_size: int, param: int, shape: tuple,
                        method: int, version: bool):
    matrix = np.zeros(shape=shape, dtype=int)
    blocks = bind_matrix(matrix, block_size)
    m, n = blocks.shape[:2]
    pointer = 0
    for i in range(m):
        for j in range(n):
            block, pointer = decode_block(encoded_string, block_size, param, pointer, method, version)
            blocks[i, j] = block
    return blocks


def decode_block(encoded_string: str, block_size: int, param: int, pointer: int, method: int, version: bool):
    vector = np.zeros(block_size ** 2, dtype=int)
    if version == IMPROVED_VERSION:
        furthest_nonzero_index, pointer = decode_symbol(encoded_string, pointer, method, False, param)
    else:
        num_of_bits = np.ceil(np.log2(block_size ** 2)).astype(int)
        furthest_nonzero_index = int(encoded_string[pointer: pointer + num_of_bits], 2)
        pointer += num_of_bits
        
    for i in range(furthest_nonzero_index + 1):
        vector[i], pointer = decode_symbol(encoded_string, pointer, method, True, param)
    block = reverse_zigzag_order(vector)
    return block, pointer


#############################################################
################### Calculate Length ########################
#############################################################


def calculate_encoding_length(encoded_blocks, param, method=EXPONENTIAL_GOLOMB, version=NORMAL_VERSION):
    length = 0
    d = encoded_blocks.shape[2]
    num_of_bits = np.ceil(np.log2(d ** 2)).astype(int)
    nonzero_indices = []
    for i in range(encoded_blocks.shape[0]):
        for j in range(encoded_blocks.shape[1]):
            block = encoded_blocks[i, j]
            zigzag_vector = zigzag_order(block)
            indices = np.nonzero(zigzag_vector)[0]
            last_nonzero_index = 0 if len(indices) == 0 else indices[-1]
            nonzero_indices.append(last_nonzero_index)
            length += np.sum(calculate_lengths(zigzag_vector[:last_nonzero_index + 1],
                                               method=method,
                                               is_signed=True,
                                               param=param))
    if version == IMPROVED_VERSION:
        length += np.sum(calculate_lengths(np.array(nonzero_indices), method, False, param))
    else:
        length += num_of_bits * encoded_blocks.shape[0] * encoded_blocks.shape[1]
    return length


def calculate_best_length(blocks, version=NORMAL_VERSION):
    k_values = list(range(1, 11))
    m_values = list(range(1, 11))
    best_length = np.inf
    best_params = None
    for k in k_values:
        length = calculate_encoding_length(blocks, k, method=EXPONENTIAL_GOLOMB, version=version)
        if length < best_length:
            best_length = length
            best_params = f'exponential-golomb: {k}'
    for m in m_values:
        length = calculate_encoding_length(blocks, m, method=GOLOMB_RICE, version=version)
        if length < best_length:
            best_length = length
            best_params = f'golomb-rice: {m}'
    return best_length, best_params


def find_best_lengths(image, block_sizes, deltas, version=NORMAL_VERSION, dpcm=False):
    len_results = np.zeros(shape=(len(block_sizes), len(deltas)))
    psnr_results = np.zeros(shape=(len(block_sizes), len(deltas)))
    params_results = [[None for _ in range(len(deltas))] for __ in range(len(block_sizes))]
    for i, n in enumerate(block_sizes):
        for j, delta in enumerate(tqdm(deltas)):
            blocks = encode_until_quantization(image, delta, n, dpcm=dpcm)
            length, params = calculate_best_length(blocks, version=version)
            rec = decode_from_quantization(blocks, delta, dpcm=dpcm)
            psnr_results[i, j] = calculate_psnr(image, rec)
            len_results[i, j] = length
            params_results[i][j] = params
    return len_results, psnr_results, params_results
