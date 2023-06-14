from typing import Tuple
import pandas as pd
import numpy as np
from tabulate import tabulate
from image_handler import *


# Q11 - Calculating Length #
def map_signed_value(value: int) -> int:
    """
    Maps a signed integer to an unsigned integer.

    Args:
        value: A signed integer.

    Returns:
        An unsigned integer.
    """
    if value < 0:
        value = value * -2
    elif value > 0:
        value = (value * 2) - 1
    else:
        value = 0
    return value


def unmap_signed_value(value: int) -> int:
    """
    Maps an unsigned integer to a signed integer.

    Args:
        value: An unsigned integer.

    Returns:
        A signed integer.
    """
    if value % 2 == 0:
        value = value // -2
    else:
        value = (value + 1) // 2
    return value


def calculate_golomb_code_length(x: int, m: int) -> int:
    """
    Calculates the length of the Golomb code for a given integer and divisor.

    Args:
        x: The integer to encode.
        m: The divisor for the Golomb code.

    Returns:
        The length of the Golomb code.
    """
    # Compute quotient and remainder
    q, r = divmod(x, m)
    # Compute b and the length of encoding
    b = (np.floor(np.log2(m))).astype(int)
    length = q + 1 + b
    # Compute u and add 1 to the total length
    u = 2 ** (b + 1) - m
    if r > u:
        length += 1
    return length


def calculate_exponential_golomb_length(x: int, k: int) -> int:
    """
    Calculates the length of the exponential Golomb code for a given integer and exponent.

    Args:
        x: The integer to encode.
        k: The exponent for the exponential Golomb code.

    Returns:
        The length of the exponential Golomb code.
    """
    b = np.floor(np.log2(x + 2 ** k)).astype(int)
    return 2 * b - k + 1


def calculate_lengths(matrix: np.ndarray, method: bool, is_signed: bool, param: int) -> np.ndarray:
    """
    Calculates the encoding lengths for a matrix of integers using the specified coding method and parameters.

    Args:
        matrix: A matrix of integers to encode.
        method: A boolean indicating the coding method to use. True for Golomb-Rice, False for exponential Golomb.
        is_signed: A boolean indicating whether to use signed or unsigned integers.
        param: An integer parameter for the specified coding method.

    Returns:
        A matrix of encoding lengths.
    """
    # Determine which coding method to use
    if method == GOLOMB_RICE:
        coding_func = calculate_golomb_code_length
    elif method == EXPONENTIAL_GOLOMB:
        coding_func = calculate_exponential_golomb_length
    else:
        raise ValueError('Unknown coding method')

    # Adjust for signed integers if needed
    if is_signed:
        matrix = np.where(matrix > 0, 2 * matrix - 1, -2 * matrix)

    return coding_func(matrix, param)


# Q12 - Optimizing compression parameters #
def optimize_compression_parameters(image: np.ndarray) -> np.ndarray:
    """
    Calculates compression ratios for various (n,k) pairs for a given image.

    Args:
        image: The image to compress.

    Returns:
        A numpy array containing the compression ratios for each (n,k) pair.
    """
    # Define the values of n and k to search over:
    n_values = [2, 4, 6, 8, 16, 24]
    k_values = list(range(6))

    # Calculate the compression ratios for each (n,k) pair:
    compression_ratios = np.empty(shape=(len(n_values), len(k_values)), dtype=np.float32)
    for n_idx, n_value in enumerate(n_values):
        binarized_image = bind_matrix(image, n_value)
        for k_idx, k_value in enumerate(k_values):
            total_length = 0
            for i in range(binarized_image.shape[0]):
                for j in range(binarized_image.shape[1]):
                    mean_block = int(np.mean(binarized_image[i, j]))
                    block = binarized_image[i, j].astype(int) - mean_block
                    total_length += \
                        np.sum(calculate_lengths(block, method=GOLOMB_RICE,
                               is_signed=True, param=k_value)) + 8
            compression_ratios[n_idx, k_idx] = total_length

    return compression_ratios / (image.size * 8)


# Q13 - Symbol encoding function
def golomb_rice_encode(x: int, m: int) -> str:
    """
    Encodes an integer x using the Golomb-Rice coding scheme with a parameter m.
    Returns the binary string representation of the encoded symbol.
    """
    encoded_symbol = ""

    # Compute quotient and remainder
    q, r = divmod(x, m)

    # Representation of q using the unary code
    encoded_symbol += '0' * q + '1'

    # Representation of r using the truncated binary code
    # Compute b and u
    b = int(np.floor(np.log2(m)))
    u = 2 ** (b + 1) - m

    if r < u:
        encoded_symbol += np.binary_repr(r, width=b)
    else:
        encoded_symbol += np.binary_repr(r + u, width=b + 1)

    return encoded_symbol


def exponential_golomb_encode(x: int, k: int) -> str:
    """
    Encodes an integer x using the Exponential-Golomb coding scheme with a parameter k.
    Returns the binary string representation of the encoded symbol.
    """
    # Compute the binary representation of integer + 2**k - 1 and its length
    binary_repr = np.binary_repr(x + 2 ** k)
    repr_len = len(binary_repr)

    return ('0' * (repr_len - 1) + binary_repr)[k:]


def encode_symbol(binary_string: str, integer: int, method: str, is_signed: bool, param: int) -> str:
    """
    Encodes an integer using the specified method and adds the resulting bits to the given binary string.

    Args:
        binary_string (str): The binary string to append the encoded bits to.
        integer (int): The integer to encode.
        method (str): The encoding method to use. Must be either 'GOLOMB_RICE' or 'EXPONENTIAL_GOLOMB'.
        is_signed (bool): Whether the integer is signed.
        param (int): The parameter to use for the encoding method.

    Returns:
        str: The binary string with the encoded bits appended.
    """
    if is_signed:
        integer = map_signed_value(integer)
    if method == GOLOMB_RICE:
        binary_string += golomb_rice_encode(integer, param)
    elif method == EXPONENTIAL_GOLOMB:
        binary_string += exponential_golomb_encode(integer, param)
    else:
        raise ValueError('Unknown coding method')
    return binary_string


# Q14 - Symbol decoding function
def golomb_rice_decode(binary_string: str, pointer: int, m: int) -> tuple[
    int, int]:
    """
    Decode an integer from a binary string using the Golomb-Rice coding scheme.

    Args:
        binary_string (str): The binary string to decode.
        pointer (int): The starting position of the decoding.
        m (int): The parameter m used in the Golomb-Rice coding scheme.

    Returns:
        tuple[int, int]: A tuple containing the decoded integer and the new pointer position.

    Raises:
        ValueError: If the new pointer position is out of bounds.
    """
    quotient = 0
    while pointer < len(binary_string) and binary_string[pointer] == '0':
        pointer += 1
        quotient += 1
    if pointer == len(binary_string) or binary_string[pointer] != '1':
        raise ValueError('New pointer is out of bounds')
    pointer += 1
    b = int(np.floor(np.log2(m)))
    u = 2 ** (b + 1) - m
    binary_repr = binary_string[pointer:pointer + b]
    remainder = int(binary_repr, 2) if b > 0 else 0
    if remainder >= u:
        binary_repr = binary_string[pointer:pointer + b + 1]
        remainder = int(binary_repr, 2) - u
        pointer += 1
    integer = quotient * m + remainder
    return integer, pointer + b


def exponential_golomb_decode(binary_string: str, pointer: int, k: int) -> \
tuple[int, int]:
    """
    Decode an integer from a binary string using the Exponential-Golomb coding scheme.

    Args:
        binary_string (str): The binary string to decode.
        pointer (int): The starting position of the decoding.
        k (int): The parameter k used in the Exponential-Golomb coding scheme.

    Returns:
        tuple[int, int]: A tuple containing the decoded integer and the new pointer position.

    Raises:
        ValueError: If the new pointer position is out of bounds.
    """
    # Find the number of leading zero bits in the code
    num_of_bits = 0
    while pointer < len(binary_string) and binary_string[pointer] == "0":
        pointer += 1
        num_of_bits += 1
    if pointer == len(binary_string) or binary_string[pointer] != '1':
        raise ValueError('New pointer is out of bounds')
    num_of_bits += k + 1
    binary_repr = binary_string[pointer:pointer + num_of_bits]
    integer = int(binary_repr, 2) - 2 ** k
    return integer, pointer + num_of_bits


def decode_symbol(binary_string: str, pointer: int, method: bool,
                  is_signed: bool, param: int):
    if method == GOLOMB_RICE:
        decoding_func = golomb_rice_decode
    elif method == EXPONENTIAL_GOLOMB:
        decoding_func = exponential_golomb_decode
    else:
        raise ValueError('Unknown coding method')
    # If the value is signed, decode the sign
    integer, pointer = decoding_func(binary_string, pointer, param)
    if is_signed:
        integer = unmap_signed_value(integer)
    return integer, pointer


def encode_block(block: np.ndarray, k: int):
    block_rows, block_cols = block.shape[:2]
    mean_block = int(np.mean(block))
    binary_string = bin(mean_block)[2:].zfill(8)
    block = block.astype(int) - mean_block
    for i in range(block_rows):
        for j in range(block_cols):
            binary_string = encode_symbol(binary_string, block[i, j],
                                          EXPONENTIAL_GOLOMB, True, k)
    return binary_string


def encode_image(matrix: np.ndarray, n: int, k: int):
    bin_m = bind_matrix(matrix, n)
    matrix_rows, matrix_cols = matrix.shape[:2]
    binary_string = ""
    for i in range(matrix_rows):
        for j in range(matrix_cols):
            binary_string += encode_block(bin_m[i, j], k)
    return binary_string


def decode_block(binary_string: str, n: int, k: int, pointer: int) -> \
        Tuple[np.ndarray, int]:
    """
    Decode a block of size n*n encoded using exponential-golomb.
    """
    block = np.zeros(shape=(n, n), dtype=np.uint8)
    mean = int(binary_string[pointer:pointer + 8], 2)
    pointer += 8
    for i in range(n):
        for j in range(n):
            x, pointer = decode_symbol(binary_string, pointer,
                                       method=EXPONENTIAL_GOLOMB,
                                       is_signed=True, param=k)
            block[i, j] = x + mean
    return block, pointer


def decode_image(binary_string, n, k, shape):
    """
    Decode an image encoded in a binary-string to an image.
    """
    matrix = np.zeros(shape=shape, dtype=np.uint8)
    blocks = bind_matrix(matrix, n)
    block_rows, block_cols = blocks.shape[:2]
    pointer = 0
    for i in range(block_rows):
        for j in range(block_cols):
            block, pointer = decode_block(binary_string, n, k, pointer)
            blocks[i, j] = block
    matrix = unbind_matrix(blocks)
    return matrix


if __name__ == '__main__':
    im = load_image(
        r"C:/Users/yuval/PycharmProjects/Image and Video Compression/Ex1/Mona-Lisa.bmp")
