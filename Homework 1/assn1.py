import numpy as np
import cv2

def create_gaussian_kernel(kernel_size, kernel_sigma):
    """Create a Gaussian kernel given kernel size and sigma."""
    ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(kernel_sigma))
    return kernel / np.sum(kernel)

def apply_filter(image, kernel):
    """Apply a filter to an image."""
    d = int(kernel.shape[0] / 2)
    padded_image = cv2.copyMakeBorder(image, d, d, d, d, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    filtered_image = np.zeros_like(image)

    for i in range(d, padded_image.shape[0] - d):
        for j in range(d, padded_image.shape[1] - d):
            window = padded_image[i - d:i + d + 1, j - d:j + d + 1]
            for k in range(3):  # Apply filter to each channel
                filtered_image[i - d, j - d, k] = np.sum(window[:, :, k] * kernel)

    return filtered_image


def filter_gaussian(image, kernel_size, kernel_sigma, border_type, separable):
    """Apply Gaussian filtering to an image."""
    # Create Gaussian kernel
    kernel = create_gaussian_kernel(kernel_size, kernel_sigma)

    # Apply border handling
    if border_type != cv2.BORDER_CONSTANT:
        image = cv2.copyMakeBorder(image, kernel_size, kernel_size, kernel_size, kernel_size, border_type)

    # Apply filter
    if separable:
        # Separable filtering
        kernel_1d = np.sqrt(kernel).sum(axis=0)
        filtered_image = apply_filter(image, kernel_1d.reshape(-1, 1))
        filtered_image = apply_filter(filtered_image, kernel_1d.reshape(1, -1))
    else:
        # Non-separable filtering
        filtered_image = apply_filter(image, kernel)

    return filtered_image

for i in range(1, 4):
    image = cv2.imread(f'color{i}.jpg')

    filtered_img = filter_gaussian(image, 5, 3.0, cv2.BORDER_CONSTANT, False)

    cv2.imwrite(f'filtered_color{i}.jpg', filtered_img)

def equalizeHistogramGrayscale(image):
    """Equalize the histogram of a grayscale image."""
    # Calculate histogram
    hist, _ = np.histogram(image.flatten(), 256, [0, 256])

    # Calculate cumulative distribution function
    cdf = hist.cumsum()
    # normalized version of cumulative distribution function
    # cdf_normalized = cdf * hist.max() / cdf.max() 
    
    # Normalize the CDF
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    # Apply the equalization
    equalized_image = cdf[image]

    return equalized_image

def equalizeHistogramColor(image):
    """Equalize the histogram of a color image."""
    # Split the image into its color channels
    channels = cv2.split(image)

    # Equalize each channel
    equalized_channels = [equalizeHistogramGrayscale(channel) for channel in channels]

    # Merge the channels back together
    equalized_image = cv2.merge(equalized_channels)

    return equalized_image


for i in range(1, 4):
    gray_image = cv2.imread(f'gray{i}.jpg', cv2.IMREAD_GRAYSCALE)
    equalized_gray = equalizeHistogramGrayscale(gray_image)

    cv2.imwrite(f'equalized_gray{i}.jpg', equalized_gray)

for i in range(1, 4):
    color_image = cv2.imread(f'color{i}.jpg')
    equalized_color = equalizeHistogramColor(color_image)

    cv2.imwrite(f'equalized_color{i}.jpg', equalized_color)