from JPEG_compressor import *
from quantizer import *


def test_until_dct(matrix: np.ndarray, block_size: int):
    """
    Test the encoding and decoding process until the DCT stage.

    Args:
        matrix: The matrix representing the image.
        block_size: The block size for encoding.
    """
    encoded_coeffs = encode_until_dct(matrix, block_size)
    reconstructed_image = decode_from_dct(encoded_coeffs).astype(np.uint8)

    imageio.imsave("Plots/reconstructed_image.jpg", reconstructed_image)

    # Compare the reconstructed image with the original image
    mse = np.mean((matrix - reconstructed_image) ** 2)
    if mse < 0.5:
        print("The reconstructed picture is identical to the original.")
    else:
        print("The reconstructed picture is not identical to the original.")


def test_entropy_stage(image: np.ndarray):
    """
    Verify that the entropy encoding stage is reversible, i.e., the indices given to the encoder
    are returned by the decoder.

    Args:
        image: The image to test.
    """
    d = 8
    delta = .2
    k = 5
    enc_blocks = encode_until_quantization(image, delta, d)
    enc_string = encode_image_blocks(enc_blocks, d, k, method=GOLOMB_RICE, version=NORMAL_VERSION)
    dec_blocks = decode_image_blocks(enc_string, d, k, image.shape, method=GOLOMB_RICE, version=NORMAL_VERSION)
    assert (enc_blocks == dec_blocks).all()


def test_length_calculation(image: np.ndarray):
    """
    Test the calculation of encoding length.

    Args:
        image: The image to test.
    """
    block_size = 8
    delta = 0.2
    k = 5
    enc_blocks = encode_until_quantization(image, delta, block_size)
    for version in [NORMAL_VERSION, IMPROVED_VERSION]:
        length = calculate_encoding_length(enc_blocks, k, method=EXPONENTIAL_GOLOMB, version=version)
        encoded_image = encode_image_blocks(enc_blocks, block_size, k, method=EXPONENTIAL_GOLOMB, version=version)
    actual_length = len(encoded_image)
    assert actual_length == length
