# def until_dct_test(matrix):
#     # Encode the image
#     encoded_coeffs = until_dct_encoder(matrix)
#
#     # Decode the coefficients
#     reconstructed_image = until_dct_decoder(encoded_coeffs)
#
#     imageio.imsave("reconstructed_image.jpg", reconstructed_image)
#
#     # Compare the reconstructed image with the original image
#     mse = np.mean((matrix - reconstructed_image) ** 2)
#     if mse < 0.5:
#         print("The reconstructed picture is identical to the original.")
#     else:
#         print("The reconstructed picture is not identical to the original.")
#
#
# def quantization_error_test(matrix, delta, block_size):
#     # Encode the image with quantization
#     encoded_coeffs = until_quantization_encoder(matrix, delta)
#
#     # Decode the coefficients with quantization
#     reconstructed_image = until_quantization_decoder(encoded_coeffs, delta)
#
#     # Calculate PSNR between original and reconstructed image
#     psnr_value = calculate_psnr(matrix, reconstructed_image, 8)
#
#     return psnr_value



# def perform_dct(matrix):
#     blocks = bind_matrix(matrix, 8)
#     blocks = scale_matrix(blocks, 8)
#     idct_check = np.zeros_like(blocks)
#     for i in range(blocks.shape[0]):
#         for j in range(blocks.shape[1]):
#             block = blocks[i][j]
#             real_dct = scipy.fft.idct(scipy.fft.dct(block, type=2))
#             my_dct = idct_2d(dct_2d(block))
#             idct_check[i][j] = my_dct
#             # print("Hi\n")
#     return unbind_matrix(blocks)

# delta_values = [8, 16]
# # Calculate PSNR for each delta value
# psnr_values = []
# for delta in delta_values:
#     # Encode the image with quantization
#     quantized_coeffs = until_quantization_encoder(matrix, delta)
#
#     # Decode the quantized coefficients
#     reconstructed_image = until_quantization_decoder(quantized_coeffs, delta)
#
#     # Calculate PSNR
#     psnr = calculate_psnr(original_image, reconstructed_image, 8)
#     psnr_values.append(psnr)
#
# # Plot the PSNR values
# plt.plot(delta_values, psnr_values)
# plt.xlabel("Delta (Quantization Level)")
# plt.ylabel("PSNR (dB)")
# plt.title("PSNR vs Delta (Block Size: {})".format(block_size))
# plt.show()


# plot_psnr_vs_compression_rate(r"C:/Users/yuval/PycharmProjects/Image and Video Compression/Ex2/Mona-Lisa.bmp")
# print(perform_dct(im))
# x = np.linspace(-10, 10, 1000)
# delta = 0.5
#
# # Quantization
# q = quantize(x, delta)
#
# # Inverse quantization
# inv_q = inverse_quantize(q, delta)
#
# print("Hi")

# block = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#
# vector = zigzag_order(block)
# print(vector)

# new_block = inverse_zigzag_order(vector)
# print(new_block)
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]
