import numpy as np
import math
import cv2
from numpy.fft import fft2, ifft2, fftshift, ifftshift

def create_ideal_lowpass_filter(shape, radius):
    """Create an ideal lowpass filter with a given radius."""
    rows, cols = shape[:2]
    center_row, center_col = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    mask = (x - center_col)**2 + (y - center_row)**2 <= radius**2
    return mask

def apply_filter_in_frequency_domain_with_padding(image_channel, radius, padding):
    """Apply ideal lowpass filter to an image channel in the frequency domain with padding."""
    padded_channel = np.pad(image_channel, ((padding, padding), (padding, padding)), mode='reflect')

    f_transform = fft2(padded_channel)
    f_transform_shifted = fftshift(f_transform)

    lowpass_filter = create_ideal_lowpass_filter(padded_channel.shape, radius)

    filtered_transform = f_transform_shifted * lowpass_filter

    f_ishift = ifftshift(filtered_transform)
    img_back = ifft2(f_ishift)
    img_back = np.abs(img_back)

    img_back = img_back[padding:-padding, padding:-padding]

    return img_back

def ideal_lowpass_filter_rgb_with_padding(image, radius, padding):
    """Apply ideal lowpass filter to an RGB image with padding."""
    channels = cv2.split(image)

    filtered_channels = [apply_filter_in_frequency_domain_with_padding(channel, radius, padding) for channel in channels]

    filtered_image = cv2.merge(filtered_channels)

    filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return filtered_image


def calculate_dynamic_parameters(image_shape, radius_ratio=0.2, padding_ratio=0.5):
    """Calculate dynamic radius and padding based on image shape and ratios."""
    rows, cols = image_shape[:2]
    diagonal_length = np.sqrt(rows**2 + cols**2)
    radius = int(diagonal_length * radius_ratio)
    padding = int(radius * padding_ratio)
    return radius, padding

def ideal_lowpass_filter_rgb_dynamic(image):
    """Apply ideal lowpass filter to an RGB image with dynamic radius and padding."""
    radius, padding = calculate_dynamic_parameters(image.shape)

    # Rest of the processing is same as before
    filtered_image = ideal_lowpass_filter_rgb_with_padding(image, radius, padding)
    return filtered_image



def create_gaussian_lowpass_filter(shape, sigma):
    """Create a Gaussian lowpass filter with a given sigma."""
    rows, cols = shape[:2]
    center_row, center_col = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    distance_squared = (x - center_col)**2 + (y - center_row)**2
    gaussian_filter = np.exp(-distance_squared / (2 * sigma**2))
    return gaussian_filter

def apply_gaussian_filter_in_frequency_domain(image_channel, sigma, padding):
    """Apply Gaussian lowpass filter to an image channel in the frequency domain with padding."""
    padded_channel = np.pad(image_channel, ((padding, padding), (padding, padding)), mode='reflect')

    f_transform = fft2(padded_channel)
    f_transform_shifted = fftshift(f_transform)

    gaussian_filter = create_gaussian_lowpass_filter(padded_channel.shape, sigma)

    filtered_transform = f_transform_shifted * gaussian_filter

    f_ishift = ifftshift(filtered_transform)
    img_back = ifft2(f_ishift)
    img_back = np.abs(img_back)

    img_back = img_back[padding:-padding, padding:-padding]

    return img_back

def gaussian_lowpass_filter_rgb_with_padding(image, sigma, padding):
    """Apply Gaussian lowpass filter to an RGB image with padding."""
    channels = cv2.split(image)

    filtered_channels = [apply_gaussian_filter_in_frequency_domain(channel, sigma, padding) for channel in channels]

    filtered_image = cv2.merge(filtered_channels)

    filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return filtered_image


def unsharp_masking(image, alpha, filter_size, domain, border_type, sigma):
    """Apply unsharp masking to an image in the specified domain."""
    padding = math.floor(filter_size / 2)
    expanded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, border_type)

    def gauss2d(shape, sigma):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m+1, -n:n+1]
        h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h / h.sum()

    flt = gauss2d((filter_size, filter_size), sigma)

    output_image = np.zeros_like(image)
    if domain == 'frequency':
        flt_f = np.fft.fft2(flt, s=expanded_image.shape[:2])
        flt_f = np.fft.fftshift(flt_f)

        for i, c in enumerate(cv2.split(expanded_image)):
            img_f = np.fft.fft2(c)
            img_flt_f = img_f + alpha * (img_f - flt_f * img_f)
            img_flt = np.real(np.fft.ifft2(img_flt_f))
            output_image[..., i] = img_flt[padding:-padding, padding:-padding]
    else: # Spatial Domain
        gaussian_kernel = gauss2d((filter_size, filter_size), sigma)
        blurred_image = cv2.filter2D(image, -1, gaussian_kernel, borderType=border_type)
        mask = image - blurred_image
        output_image = cv2.addWeighted(image, 1.0 + alpha, mask, -alpha, 0)

    output_image = cv2.normalize(output_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return output_image

def main():
    for i in range(1, 4):
        # problem 1
        image_path = f'color{i}.jpg'
        image = cv2.imread(image_path)
        filtered1_img = ideal_lowpass_filter_rgb_dynamic(image)
        cv2.imwrite(f'filtered_image_dynamic{i}.jpg', filtered1_img)

        # problem 2
        sigma = 10
        padding = 50
        filtered2_img = gaussian_lowpass_filter_rgb_with_padding(image, sigma, padding)
        cv2.imwrite(f'gaussian_filtered_image{i}.jpg', filtered2_img)
        
        # problem 3
        sharpened_image = unsharp_masking(image, alpha=2.0, filter_size=77, domain='spatial', border_type=1, sigma=1.0)
        cv2.imwrite(f'sharpened_image{i}.jpg', sharpened_image)


if __name__ == "__main__":
    main()