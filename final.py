import numpy as np
from skimage import io, img_as_float, img_as_ubyte
from tqdm import tqdm

# Relleno de ceros para convertir a potencias de 2
def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()

def pad_to_power_of_2(image):
    # Verificar si la imagen es en escala de grises (2D) o color (3D)
    if len(image.shape) == 2:
        # Agregar un tercer eje para simular un canal
        image = image[:, :, None]

    rows, cols, channels = image.shape

    # Encuentra la próxima potencia de 2 para filas y columnas
    new_rows = 2**np.ceil(np.log2(rows)).astype(int)
    new_cols = 2**np.ceil(np.log2(cols)).astype(int)

    # Realizar el padding
    padded_image = np.zeros((new_rows, new_cols, channels))
    padded_image[:rows, :cols, :] = image

    # Asegurarse de devolver la imagen en su formato original (2D o 3D)
    if channels == 1:
        padded_image = padded_image[:, :, 0]  # Convertir de nuevo a 2D si es necesario

    return padded_image, (rows, cols)

def fft_manual(x):
    N = x.shape[0]
    if N <= 1:
        return x
    even = fft_manual(x[0::2])
    odd = fft_manual(x[1::2])
    factor = np.exp(-2j * np.pi * np.arange(N) / N)
    T = factor[:N // 2] * odd
    return np.concatenate([even + T, even - T])

def fft2_manual(image):
    # FFT de cada fila
    padded_image, original_shape = pad_to_power_of_2(image)
    rows_fft = np.array([fft_manual(row) for row in tqdm(padded_image, desc="FFT Rows")])
    # FFT de cada columna
    cols_fft = np.array([fft_manual(col) for col in tqdm(rows_fft.T, desc="FFT Columns")]).T
    return cols_fft[:image.shape[0], :image.shape[1]]

def ifft_manual(x):
    x_conj = np.conjugate(x)
    result = fft_manual(x_conj)
    return np.conjugate(result) / x.shape[0]

def ifft2_manual(spectrum):
    # IFFT de cada fila
    rows_ifft = np.array([ifft_manual(row) for row in tqdm(spectrum, desc="IFFT Rows")])
    # IFFT de cada columna
    cols_ifft = np.array([ifft_manual(col) for col in tqdm(rows_ifft.T, desc="IFFT Columns")]).T
    return cols_ifft[:spectrum.shape[0], :spectrum.shape[1]]

def compress_channel_manual(channel, compression_rate):
    """ Apply manual FFT, zero out coefficients, and apply manual inverse FFT. """
    # Perform FFT on the 2D channel
    transformed = fft2_manual(channel)
    
    # Flatten the 2D FFT result and sort by magnitude
    flat = np.abs(transformed).flatten()
    threshold = np.percentile(flat, (1 - compression_rate) * 100)

    # Zero out the smallest coefficients
    transformed[np.abs(transformed) < threshold] = 0

    # Perform inverse FFT
    compressed_channel = ifft2_manual(transformed).real
    return compressed_channel

def compress_image_manual(image, compression_rates):
    """ Compress an image using manual FFT at multiple compression rates. """
    padded_image, original_shape = pad_to_power_of_2(image)
    h, w = original_shape
    compressed_images = []
    
    for rate in compression_rates:
        compressed_image = np.zeros_like(padded_image)
        for channel in range(3):
            compressed_image[..., channel] = compress_channel_manual(padded_image[..., channel], rate)
        compressed_images.append(compressed_image[:h, :w, :])
    
    return compressed_images

# Load and preprocess the image
image_path = 'IMG/ciudad.jpg'
image = img_as_float(io.imread(image_path))  # Normalizar a [0, 1]

# Set compression rates
compression_rates = [0.5, 0.8, 0.95]

# Compress the image
compressed_images = compress_image_manual(image, compression_rates)

# Save the original and compressed images
io.imsave('original_image.jpg', img_as_ubyte(image))

for i, rate in enumerate(compression_rates):
    compressed_image_uint8 = img_as_ubyte(np.clip(compressed_images[i], 0, 1))
    io.imsave(f'IMG/compressed_image_{int(rate * 100)}.jpg', compressed_image_uint8)
