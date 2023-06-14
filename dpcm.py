from dct import *


def apply_dpcm_encoding(encoded_blocks, delta):
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


def apply_dpcm_decoding(blocks):
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


def encode_until_dpcm(image, delta, block_size):
    encoded_blocks = encode_until_dct(image, block_size)
    return apply_dpcm_encoding(encoded_blocks, delta)


def decode_from_dpcm(blocks, resolution):
    blocks = apply_dpcm_decoding(blocks)
    image = decode_from_dct(blocks, resolution)
    return image
