from typing import Tuple
from image_operations import *

GOLOMB_RICE = 0
EXPONENTIAL_GOLOMB = 1


#############################################################
######################## Encoder ############################
#############################################################
def encode_symbol(binary_string: str, integer: int, method: int, is_signed: bool, param: int) -> str:
    """
    Encodes an integer using the specified method and adds the resulting bits to the given binary string.

    Args:
        binary_string: The binary string to append the encoded bits to.
        integer: The integer to encode.
        method: The encoding method to use. Must be either `GOLOMB_RICE` or `EXPONENTIAL_GOLOMB`.
        is_signed: Whether the integer is signed.
        param: The parameter to use for the encoding method.

    Returns:
        The binary string with the encoded bits appended.
    """
    if is_signed:
        integer = 2 * integer - 1 if integer > 0 else -2 * integer
    if method == GOLOMB_RICE:
        binary_string += golomb_rice_encode(integer, param)
    elif method == EXPONENTIAL_GOLOMB:
        binary_string += exponential_golomb_encode(integer, param)
    else:
        raise ValueError('Unknown coding method')
    return binary_string


def golomb_rice_encode(x: int, m: int) -> str:
    """
    Encode an integer using the Golomb-Rice coding scheme with a parameter m.

    Args:
        x: The integer to encode.
        m: The parameter for Golomb-Rice encoding.

    Returns:
        The binary string representation of the encoded symbol.
    """
    q, r = divmod(x, m)
    encoded_symbol = '0' * q + '1'
    encoded_symbol += truncated_binary_encode(r, m)
    return encoded_symbol


def exponential_golomb_encode(x: int, k: int) -> str:
    """
    Encode an integer using the Exponential-Golomb coding scheme with a parameter k.

    Args:
        x: The integer to encode.
        k: The parameter for Exponential-Golomb encoding.

    Returns:
        The binary string representation of the encoded symbol.
    """
    # Compute the binary representation of integer + 2**k - 1 and its length
    binary_x = np.binary_repr(x + 2 ** k)
    x_repr_len = len(binary_x)
    leading_zeros = "0" * (x_repr_len - k - 1)
    return leading_zeros + binary_x


def truncated_binary_encode(r: int, m: int) -> str:
    """
    Encodes an integer using truncated binary encoding with a maximum value.

    Args:
        r: The integer to encode.
        m: The maximum value for the encoding.

    Returns:
        The binary string representation of the encoded symbol.
    """
    b = (np.floor(np.log2(m))).astype(int)
    u = 2 ** (b + 1) - m
    truncated_string = bin(r)[2:].zfill(b) if r < u else bin(r + u)[2:]
    return truncated_string


def encode_block(block: np.ndarray, k: int) -> str:
    """
    Encodes a block of integers using exponential Golomb encoding with a parameter.

    Args:
        block: The block of integers to encode.
        k: The parameter for exponential Golomb encoding.

    Returns:
        The binary string representation of the encoded block.
    """
    block_rows, block_cols = block.shape[:2]
    mean_block = int(np.mean(block))
    binary_string = bin(mean_block)[2:].zfill(8)
    block = block.astype(int) - mean_block
    for i in range(block_rows):
        for j in range(block_cols):
            binary_string = encode_symbol(binary_string, block[i, j], EXPONENTIAL_GOLOMB, True, k)
    return binary_string


def encode_image(matrix: np.ndarray, n: int, k: int) -> str:
    """
    Encodes an image represented as a matrix of integers using block-based encoding with exponential Golomb.

    Args:
        matrix: The matrix representing the image.
        n: The size of each block.
        k: The parameter for exponential Golomb encoding.

    Returns:
        The binary string representation of the encoded image.
    """
    bin_matrix = bind_matrix(matrix, n)
    matrix_rows, matrix_cols = matrix.shape[:2]
    binary_string = ""
    for i in range(matrix_rows):
        for j in range(matrix_cols):
            binary_string += encode_block(bin_matrix[i, j], k)
    return binary_string


def decode_symbol(binary_string: str, pointer: int, method: int, is_signed: bool, param: int) -> Tuple[int, int]:
    """
    Decodes an integer symbol from a binary string using the specified decoding method and parameters.

    Args:
        binary_string: The binary string to decode from.
        pointer: The current position in the binary string.
        method: The decoding method to use (GOLOMB_RICE or EXPONENTIAL_GOLOMB).
        is_signed: Whether the symbol is signed.
        param: The parameter for the decoding method.

    Returns:
        A tuple containing the decoded integer and the updated pointer position.
    """
    if method == GOLOMB_RICE:
        decoding_func = golomb_rice_decode
    elif method == EXPONENTIAL_GOLOMB:
        decoding_func = exponential_golomb_decode
    else:
        raise ValueError('Unknown coding method')
    integer, pointer = decoding_func(binary_string, pointer, param)
    if is_signed:
        integer = integer // -2 if integer % 2 == 0 else (integer + 1) // 2
    return integer, pointer


#############################################################
######################## Decoder ############################
#############################################################

def golomb_rice_decode(binary_string: str, pointer: int, m: int) -> Tuple[int, int]:
    """
    Decode an integer from a binary string using the Golomb-Rice coding scheme.

    Args:
        binary_string: The binary string to decode from.
        pointer: The current position in the binary string.
        m: The parameter for Golomb-Rice decoding.

    Returns:
        A tuple containing the decoded integer and the updated pointer position.
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
    r = int(binary_repr, 2) if b > 0 else 0
    if r >= u:
        binary_repr = binary_string[pointer:pointer + b + 1]
        r = int(binary_repr, 2) - u
        pointer += 1
    integer = quotient * m + r
    return integer, pointer + b


def exponential_golomb_decode(binary_string: str, pointer: int, k: int) -> tuple[int, int]:
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
    while pointer < len(binary_string) and binary_string[pointer] == '0':
        pointer += 1
        num_of_bits += 1
    if pointer == len(binary_string) or binary_string[pointer] != '1':
        raise ValueError('New pointer is out of bounds')
    num_of_bits += k + 1
    binary_repr = binary_string[pointer:pointer + num_of_bits]
    integer = int(binary_repr, 2) - 2 ** k
    return integer, pointer + num_of_bits


def decode_block(binary_string: str, n: int, k: int, pointer: int) -> Tuple[np.ndarray, int]:
    """
    Decode a block of size n*n encoded using exponential Golomb encoding.

    Args:
        binary_string: The binary string to decode from.
        n: The size of the block.
        k: The parameter for exponential Golomb encoding.
        pointer: The current position in the binary string.

    Returns:
        A tuple containing the decoded block and the updated pointer position.
    """
    block = np.zeros(shape=(n, n), dtype=np.uint8)
    mean = int(binary_string[pointer:pointer + 8], 2)
    pointer += 8
    for i in range(n):
        for j in range(n):
            x, pointer = decode_symbol(binary_string, pointer, method=EXPONENTIAL_GOLOMB, is_signed=True, param=k)
            block[i, j] = x + mean
    return block, pointer


def decode_image(binary_string: str, n: int, k: int, shape: Tuple[int, int]) -> np.ndarray:
    """
    Decode an image encoded in a binary string to an image matrix.

    Args:
        binary_string: The binary string representation of the encoded image.
        n: The size of each block.
        k: The parameter for exponential Golomb encoding.
        shape: The shape of the decoded image matrix.

    Returns:
        The decoded image matrix.
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


#############################################################
################### Calculate Length ########################
#############################################################
def calculate_lengths(matrix: np.ndarray, method: int, is_signed: bool, param: int) -> int:
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


def calculate_golomb_code_length(x: np.ndarray, m: int) -> int:
    """
    Calculates the length of the Golomb code for a given integer and divisor.

    Args:
        x: The integer to encode.
        m: The divisor for the Golomb code.

    Returns:
        The length of the Golomb code.
    """
    q, r = divmod(x, m)
    b = (np.floor(np.log2(m))).astype(int)
    u = 2 ** (b + 1) - m
    truncated_binary_length = np.where(r < u, b, b + 1)
    return q + 1 + truncated_binary_length


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


if __name__ == '__main__':
    im = load_image(r"Mona-Lisa.bmp")


