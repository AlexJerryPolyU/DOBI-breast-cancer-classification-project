"""
image_processing.py - Part of fNIR Base Model.
"""

from locale import normalize

import pandas as pd
from scipy.ndimage import uniform_filter
from skimage.filters import difference_of_gaussians, gabor, roberts
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
from skimage import restoration
from skimage.restoration import denoise_nl_means, estimate_sigma, denoise_tv_chambolle, denoise_bilateral, denoise_wavelet, inpaint
from skimage import filters, morphology
from skimage.feature import hessian_matrix, hessian_matrix_eigvals, canny
from scipy.ndimage import gaussian_laplace
import cv2
import pywt
from skimage.restoration import denoise_tv_bregman

class ImageProcessor:
    def __init__(self, interpolation_method=0, denoising_method=0, normalization_method=2, enhancement_method=0, edge_detection_method=0):
        self.interpolation_method = interpolation_method
        self.denoising_method = denoising_method
        self.normalization_method = normalization_method
        self.enhancement_method = enhancement_method
        self.edge_detection_method = edge_detection_method
        print('interpolation_method:', self.interpolation_method)
        print('denoising_method:', self.denoising_method)
        print('normalization_method:', self.normalization_method)
        print('enhancement_method:', self.enhancement_method)
        print('edge_detection_method:', self.edge_detection_method)

    def resize(self, image, target_depth, target_height=None, target_width=None, single_layer=False):
        """Resize a 3D image to target dimensions using selected interpolation method.

        Args:
        - image: 输入图像
        - target_depth: 目标深度
        - target_height: 目标高度 (可选)
        - target_width: 目标宽度 (可选)
        - single_layer: 是否为单层图像，默认为 False
        """
        if single_layer:
            # 对单层图像进行处理
            height, width, current_depth = image.shape
            resized_image = resize(image, (target_height, target_width, target_depth), mode='reflect', anti_aliasing=True,
                                   order=self.interpolation_method)
            # 转置为 (channels, depth, height, width)
            resized_image = resized_image[:, :, np.newaxis, :]
            resized_image = resized_image.transpose(3, 2, 0, 1)
            return resized_image
        else:
            # 对多层图像进行处理
            height, width, current_depth, channels = image.shape
            # 转置为 (height, width, channels, depth) 后再调整大小
            image_transposed = np.transpose(image, (0, 1, 3, 2))  # (height, width, channels, depth)
            resized_image = resize(image_transposed, (target_height, target_width, channels, target_depth), mode='reflect',
                                   anti_aliasing=True, order=self.interpolation_method)
            # 转置回 (channels, depth, height, width)
            resized_image = resized_image.transpose(3, 2, 0, 1)
            return resized_image

    def resize_MC(self, image, target_depth, target_height=None, target_width=None):
        # 对多层图像进行处理
        height, width, channels, current_depth  = image.shape
        # 转置为 (height, width, channels, depth) 后再调整大小
        resized_image = resize(image,(target_height, target_width, channels, target_depth), mode='reflect',
                                   anti_aliasing=True, order=self.interpolation_method)
        # 转置回 (channels, depth, height, width)
        resized_image = resized_image.transpose(3, 2, 0, 1)
        return resized_image

    # 图像去噪
    # def denoise(self, image):
    #     """Apply denoising based on selected method."""
    #     if self.denoising_method == 0:
    #         return image
    #     methods = [
    #         self.non_local_means_denoising,  # 1: Non-Local Means Denoising
    #         self.gaussian_filter_smoothing,  # 2: Gaussian Smoothing
    #         self.median_filtering,  # 3: Median Filtering
    #         self.wiener_deconvolution,  # 4: Wiener Filter Deconvolution
    #         self.richardson_lucy_deconvolution,  # 5: Richardson-Lucy Deconvolution
    #         self.tv_denoising,  # 6: Total Variation Denoising
    #         self.bilateral_filtering,  # 7: Bilateral Filtering
    #         self.wavelet_denoising,  # 8: Wavelet Denoising
    #         self.inpainting,  # 9: Inpainting for Missing Pixels
    #         self.isotropic_diffusion,  # 10: Isotropic Diffusion
    #         self.anisotropic_diffusion  # 11: Anisotropic Diffusion
    #     ]
    #     # Apply the selected denoising method and display the results
    #     denoised_image = methods[self.denoising_method-1](image)
    #     self.show_image_comparison(image, denoised_image, f'Denoising Method {self.denoising_method}')
    #     return denoised_image

    def show_image_comparison(self, original, processed, title):
        """Display a side-by-side comparison of the original and processed images."""
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(original, cmap="gray")

        plt.subplot(1, 2, 2)
        plt.title(title)
        plt.imshow(processed, cmap="gray")
        plt.show()

    # def non_local_means_denoising(self, image):
    #     """Non-Local Means Denoising."""
    #     sigma_est = np.mean(estimate_sigma(image))  # 移除了 multichannel 参数
    #     return denoise_nl_means(image, h=1.15 * sigma_est, fast_mode=True, patch_size=5, patch_distance=6)

    def non_local_means_denoising(self, image):
        """
        Apply Non-Local Means Denoising to 23x1x102x128 data.
        Parameters:
            image: np.ndarray of shape (23, 1, 102, 128)
        Returns:
            Denoised image of the same shape as input.
        """
        # Ensure input is in the correct shape
        # print(image.shape)
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError("Input must be of shape (23, 1, 102, 128)")

        # Initialize output array
        denoised_images = np.zeros_like(image)

        # Process each frame
        for i in range(image.shape[0]):
            single_channel_image = image[i, 0]  # Extract single-channel frame (102x128)

            # Estimate noise level
            sigma_est = np.mean(estimate_sigma(single_channel_image, channel_axis=None))

            # Apply Non-Local Means Denoising
            denoised_frame = denoise_nl_means(
                single_channel_image,
                h=1.15 * sigma_est,
                fast_mode=True,
                patch_size=5,
                patch_distance=6
            )

            # Store the denoised frame back with proper shape
            denoised_images[i, 0] = denoised_frame

        return denoised_images

    def gaussian_filter_smoothing(self, image, sigma=1):
        """Gaussian Filter Smoothing."""
        return filters.gaussian(image, sigma=sigma)

    def median_filtering(self, image):
        """Median Filtering for 23x1x102x128 data."""
        # Initialize the output array to store the denoised frames
        denoised_image = np.zeros_like(image)

        # Apply median filtering to each frame (23, 1, 102, 128)
        for i in range(image.shape[0]):
            single_frame = image[i, 0, :, :]  # Extract the 102x128 frame (single channel)
            denoised_frame = filters.median(single_frame, morphology.disk(3))  # Apply median filter
            denoised_image[i, 0, :, :] = denoised_frame  # Store the denoised frame back

        return denoised_image

    def wiener_deconvolution(self, image):
        """Wiener Filter Deconvolution for 23x1x102x128 data."""
        # Initialize output array to store the denoised frames
        deconvolved_image = np.zeros_like(image)

        # Point Spread Function (PSF)
        psf = np.ones((5, 5)) / 25  # A simple average filter

        # Apply Wiener deconvolution to each frame (23, 1, 102, 128)
        for i in range(image.shape[0]):
            single_frame = image[i, 0, :, :]  # Extract the 102x128 frame (single channel)

            # Apply Wiener deconvolution to the single frame
            deconvolved_frame = restoration.wiener(single_frame, psf, balance=0.1)

            # Store the deconvolved frame back to the output array
            deconvolved_image[i, 0, :, :] = deconvolved_frame
        return deconvolved_image

    def richardson_lucy_deconvolution(self, image):
        """Richardson-Lucy Deconvolution for 23x1x102x128 data."""
        # Initialize output array to store the deconvolved frames
        deconvolved_image = np.zeros_like(image)

        # Point Spread Function (PSF)
        psf = np.ones((5, 5)) / 25  # A simple average filter

        # Apply Richardson-Lucy deconvolution to each frame (23, 1, 102, 128)
        for i in range(image.shape[0]):
            single_frame = image[i, 0, :, :]  # Extract the 102x128 frame (single channel)

            # Corrected: Pass iterations as a positional argument
            deconvolved_frame = restoration.richardson_lucy(single_frame, psf, 30)  # 30 is the number of iterations

            # Store the deconvolved frame back to the output array
            deconvolved_image[i, 0, :, :] = deconvolved_frame

        return deconvolved_image

    def tv_denoising(self, image, weight=0.1):
        """Total Variation Denoising (TV Denoising)."""
        return denoise_tv_chambolle(image, weight=weight)

    def bilateral_filtering(self, image):
        """Bilateral Filtering for 23x1x102x128 data."""
        # Remove the unnecessary single channel dimension
        image = image[:, 0, :, :]  # Shape becomes (23, 102, 128)

        # Apply bilateral filtering to each frame in the 3D stack
        filtered_image = np.array([
            denoise_bilateral(frame, sigma_color=0.05, sigma_spatial=15)
            for frame in image
        ])

        # Add the single channel dimension back to match the original shape
        return filtered_image[:, np.newaxis, :, :]  # Shape back to (23, 1, 102, 128)

    def wavelet_denoising(self, data):
        """
        Apply Wavelet Denoising to each frame in the data.
        Args:
            data (numpy.ndarray): Input data with shape (23, 1, 102, 128).
        Returns:
            numpy.ndarray: Denoised data with the same shape as input.
        """
        denoised_data = np.zeros_like(data)
        for i in range(data.shape[0]):  # Loop over frames
            # Remove the singleton dimension to apply denoise_wavelet
            denoised_frame = denoise_wavelet(data[i, 0])
            # Add back the singleton dimension
            denoised_data[i, 0] = denoised_frame
        return denoised_data

    def inpainting(self, image):
        """Inpainting for Missing Pixels (Biharmonic Inpainting)."""
        mask = np.zeros(image.shape)
        mask[30:50, 30:50] = 1  # Define a region to inpaint (modify as needed)
        return inpaint.inpaint_biharmonic(image, mask)

    def isotropic_diffusion(self, image, weight=10.0):
        """各向同性扩散（使用 Total Variation Bregman 进行边缘保护平滑）。"""

        # 如果是四维数组，逐张图像处理
        if len(image.shape) == 4:
            output_images = np.empty_like(image)  # 创建一个空数组来存储处理后的图像

            for i in range(image.shape[0]):
                single_image = image[i, 0]  # 获取每一张单通道图像 (102x128)
                denoised_image = denoise_tv_bregman(single_image, weight=weight)  # 去噪
                output_images[i, 0] = denoised_image  # 将去噪后的图像保存

            return output_images

        else:
            # 处理单张图像的情况
            return denoise_tv_bregman(image, weight=weight)


    def anisotropic_diffusion(self, image, weight=10):
        """各向异性扩散（使用 Total Variation Bregman 进行边缘保护平滑）。"""
        # 如果图像是四维数据，逐张图像进行处理
        if len(image.shape) == 4:
            output_images = np.empty_like(image)  # 创建一个空数组来存储处理后的图像

            for i in range(image.shape[0]):
                single_image = image[i, 0]  # 获取每一张单通道图像 (102x128)
                single_image = single_image.astype(np.float64)  # 确保为 float64 类型
                denoised_image = denoise_tv_bregman(single_image, isotropic=False, weight=weight)  # 去噪
                output_images[i, 0] = denoised_image  # 将去噪后的图像保存

            return output_images

        else:
            # 处理单张图像的情况
            image = image.astype(np.float64)  # 确保为 float64 类型
            return denoise_tv_bregman(image, isotropic=False, weight=weight)

    # 图像去噪
    def denoise(self, image):
        """Apply denoising based on selected method."""
        if self.denoising_method == 0:
            return image
        methods = [
            self.non_local_means_denoising,  # 1: Non-Local Means Denoising
            self.gaussian_filter_smoothing,  # 2: Gaussian Smoothing
            self.median_filtering,  # 3: Median Filtering
            self.wiener_deconvolution,  # 4: Wiener Filter Deconvolution
            self.richardson_lucy_deconvolution,  # 5: Richardson-Lucy Deconvolution
            self.tv_denoising,  # 6: Total Variation Denoising
            self.bilateral_filtering,  # 7: Bilateral Filtering
            self.wavelet_denoising,  # 8: Wavelet Denoising
            self.inpainting,  # 9: Inpainting for Missing Pixels
            self.isotropic_diffusion,  # 10: Isotropic Diffusion
            self.anisotropic_diffusion  # 11: Anisotropic Diffusion
        ]
        # Apply the selected denoising method and display the results
        denoised_image = methods[self.denoising_method-1](image)
        # self.show_image_comparison(image, denoised_image, f'Denoising Method {self.denoising_method}')
        return denoised_image


    # 标准化
    def normalize(self, image, names):
        """Apply normalization based on selected method."""
        if self.normalization_method == 0:
            return image  # 不做标准化
        # elif self.normalization_method == 25:
        #     return  self.mean_std_gamma_self(image, names, sheet_name, gamma_0=0.6, gamma_N=1, gamma_2=1.2)

        methods = [
            self.min_max_normalization,  # 1: Min-Max normalization
            self.mean_std_normalization,  # 2: Mean-std normalization
            self.robust_normalization, # 3：Robust normalization
            self.l2_normalization, # 4：L2 normalization
            self.log_normalization, # 5：Log normalization
            self.gamma_normalization_1, # 6：Gamma normalization (Gamma=0.5)
            self.gamma_normalization_2,  # 7：Gamma normalization (Gamma=2.2)
            self.min_max_normalization_with_clipping, # 8：Min-Max normalization with clipping
            self.quantile_normalization, # 9：Quantile normalization
            self.standardization_with_scaling, # 10：Standardization with scaling
            self.local_contrast_normalization, # 11：Local contrast normalization
            self.gamma_normalization_3,  # 12：Gamma normalization (Gamma=0.2)
            self.gamma_normalization_4,  # 13：Gamma normalization (Gamma=0.4)
            self.gamma_normalization_5,  # 14：Gamma normalization (Gamma=0.6)
            self.gamma_normalization_6,  # 15：Gamma normalization (Gamma=0.8)
            self.gamma_normalization_7,  # 16：Gamma normalization (Gamma=1.2)
            self.gamma_normalization_8,  # 17：Gamma normalization (Gamma=1.4)
            self.gamma_normalization_9,  # 18：Gamma normalization (Gamma=1.6)
            self.gamma_normalization_10,  # 19：Gamma normalization (Gamma=1.8)
            self.gamma_normalization_11,  # 20：Gamma normalization (Gamma=2.0)
            self.mean_std_gamma_preprocessing_1,  # 21：Gamma normalization (Gamma=0.2)
            self.mean_std_gamma_preprocessing_2,  # 22：Gamma normalization (Gamma=0.4)
            self.mean_std_gamma_preprocessing_3,  # 23：Gamma normalization (Gamma=0.6)
            self.mean_std_gamma_preprocessing_4,  # 24：Gamma normalization (Gamma=0.8)
        ]
        method = methods[self.normalization_method-1]
        normalize_image = method(image, names)
        return normalize_image

    def robust_normalization(self, image, names):
        median = np.median(image)
        q75, q25 = np.percentile(image, [75, 25])
        iqr = q75 - q25
        iqr = iqr if iqr != 0 else 1
        normalized = (image - median) / iqr
        return normalized

    def l2_normalization(self, image, names):
        norm = np.linalg.norm(image)
        norm = norm if norm != 0 else 1
        normalized = image / norm
        return normalized

    def log_normalization(self, image, names,epsilon=1e-6):

        # 1) Convert image to float
        image = image.astype(float)

        # 2) Find image minimum and maximum
        image_min = np.min(image)
        image_max = np.max(image)

        # 3) Check if the range is zero (to avoid division by zero)
        range_val = (image_max - image_min) + epsilon
        if np.isclose(range_val, 0.0):
            # If all pixel values are the same, return an array of zeros
            print("Warning: All pixel values are identical. Returning zeros.")
            return np.zeros_like(image, dtype=float)

        # 4) Normalize the image to [0, 1] range
        #    (shifting by image_min, then dividing by the image range)
        image_normalized = (image - image_min) / range_val

        # 5) Apply log(1 + x) to the normalized image
        #    np.log1p(x) is equivalent to np.log(1 + x), but more numerically stable
        normalized = np.log(image_normalized)
        return normalized

    def gamma_normalization_1(self, image, names,gamma=0.5):
        # Step 1: Get the image's min and max values
        image_min = np.min(image)
        image_max = np.max(image)

        # Step 2: Normalize the image to [0, 1] based on the image's min and max values
        image_normalized = (image - image_min) / (image_max - image_min)

        # Step 3: Apply gamma correction
        normalized = np.power(image_normalized, gamma)

        # Return the gamma-corrected image
        return normalized

    def gamma_normalization_2(self, image,names, gamma=2.2):
        # Step 1: Get the image's min and max values
        image_min = np.min(image)
        image_max = np.max(image)

        # Step 2: Normalize the image to [0, 1] based on the image's min and max values
        image_normalized = (image - image_min) / (image_max - image_min)

        # Step 3: Apply gamma correction
        normalized = np.power(image_normalized, gamma)

        # Return the gamma-corrected image
        return normalized

    def gamma_normalization_3(self, image, names,gamma=0.2):
        # Step 1: Get the image's min and max values
        image_min = np.min(image)
        image_max = np.max(image)

        # Step 2: Normalize the image to [0, 1] based on the image's min and max values
        image_normalized = (image - image_min) / (image_max - image_min)

        # Step 3: Apply gamma correction
        normalized = np.power(image_normalized,names, gamma)

        # Return the gamma-corrected image
        return normalized

    def gamma_normalization_4(self, image, names,gamma=0.4):
        # Step 1: Get the image's min and max values
        image_min = np.min(image)
        image_max = np.max(image)

        # Step 2: Normalize the image to [0, 1] based on the image's min and max values
        image_normalized = (image - image_min) / (image_max - image_min)

        # Step 3: Apply gamma correction
        normalized = np.power(image_normalized, gamma)

        # Return the gamma-corrected image
        return normalized

    def gamma_normalization_5(self, image, names, gamma=0.6):
        # Step 1: Get the image's min and max values
        image_min = np.min(image)
        image_max = np.max(image)

        # Step 2: Normalize the image to [0, 1] based on the image's min and max values
        image_normalized = (image - image_min) / (image_max - image_min)

        # Step 3: Apply gamma correction
        normalized = np.power(image_normalized, gamma)

        # Return the gamma-corrected image
        return normalized

    def gamma_normalization_6(self, image, names, gamma=0.8):
        # Step 1: Get the image's min and max values
        image_min = np.min(image)
        image_max = np.max(image)

        # Step 2: Normalize the image to [0, 1] based on the image's min and max values
        image_normalized = (image - image_min) / (image_max - image_min)

        # Step 3: Apply gamma correction
        normalized = np.power(image_normalized, gamma)

        # Return the gamma-corrected image
        return normalized

    def gamma_normalization_7(self, image, names, gamma=1.2):
        # Step 1: Get the image's min and max values
        image_min = np.min(image)
        image_max = np.max(image)

        # Step 2: Normalize the image to [0, 1] based on the image's min and max values
        image_normalized = (image - image_min) / (image_max - image_min)

        # Step 3: Apply gamma correction
        normalized = np.power(image_normalized, gamma)

        # Return the gamma-corrected image
        return normalized

    def gamma_normalization_8(self, image, names, gamma=1.4):
        # Step 1: Get the image's min and max values
        image_min = np.min(image)
        image_max = np.max(image)

        # Step 2: Normalize the image to [0, 1] based on the image's min and max values
        image_normalized = (image - image_min) / (image_max - image_min)

        # Step 3: Apply gamma correction
        normalized = np.power(image_normalized, gamma)

        # Return the gamma-corrected image
        return normalized

    def gamma_normalization_9(self, image, names, gamma=1.6):
        # Step 1: Get the image's min and max values
        image_min = np.min(image)
        image_max = np.max(image)

        # Step 2: Normalize the image to [0, 1] based on the image's min and max values
        image_normalized = (image - image_min) / (image_max - image_min)

        # Step 3: Apply gamma correction
        normalized = np.power(image_normalized, gamma)

        # Return the gamma-corrected image
        return normalized

    def gamma_normalization_10(self, image, names, gamma=1.8):
        # Step 1: Get the image's min and max values
        image_min = np.min(image)
        image_max = np.max(image)

        # Step 2: Normalize the image to [0, 1] based on the image's min and max values
        image_normalized = (image - image_min) / (image_max - image_min)

        # Step 3: Apply gamma correction
        normalized = np.power(image_normalized, gamma)

        # Return the gamma-corrected image
        return normalized

    def gamma_normalization_11(self, image, names, gamma=2.0):
        # Step 1: Get the image's min and max values
        image_min = np.min(image)
        image_max = np.max(image)

        # Step 2: Normalize the image to [0, 1] based on the image's min and max values
        image_normalized = (image - image_min) / (image_max - image_min)

        # Step 3: Apply gamma correction
        normalized = np.power(image_normalized, gamma)

        # Return the gamma-corrected image
        return normalized

    def mean_std_gamma_preprocessing_1(self, image, names, gamma=0.2):
        # Step 1: Mean-Std Normalization
        image_mean = np.mean(image)
        image_std = np.std(image)
        normalized_image = (image - image_mean) / image_std

        # Step 2: Shift to Positive Values (if necessary)
        shifted_image = normalized_image - np.min(normalized_image)

        # Step 3: Apply Gamma Correction
        gamma_corrected = np.power(shifted_image, gamma)

        # Step 4: Rescale Back to Mean-Std Normalization (Optional)
        normalized = (gamma_corrected - np.mean(gamma_corrected)) / np.std(gamma_corrected)

        # Return the gamma-mean_std_corrected image
        return normalized

    def mean_std_gamma_preprocessing_2(self, image, names, gamma=0.4):
        # Step 1: Mean-Std Normalization
        image_mean = np.mean(image)
        image_std = np.std(image)
        normalized_image = (image - image_mean) / image_std

        # Step 2: Shift to Positive Values (if necessary)
        shifted_image = normalized_image - np.min(normalized_image)

        # Step 3: Apply Gamma Correction
        gamma_corrected = np.power(shifted_image, gamma)

        # Step 4: Rescale Back to Mean-Std Normalization (Optional)
        normalized = (gamma_corrected - np.mean(gamma_corrected)) / np.std(gamma_corrected)

        # Return the gamma-mean_std_corrected image
        return normalized

    def mean_std_gamma_preprocessing_3(self, image, names, gamma=0.6):
        # Step 1: Mean-Std Normalization
        image_mean = np.mean(image)
        image_std = np.std(image)
        normalized_image = (image - image_mean) / image_std

        # Step 2: Shift to Positive Values (if necessary)
        shifted_image = normalized_image - np.min(normalized_image)

        # Step 3: Apply Gamma Correction
        gamma_corrected = np.power(shifted_image, gamma)

        # Step 4: Rescale Back to Mean-Std Normalization (Optional)
        normalized = (gamma_corrected - np.mean(gamma_corrected)) / np.std(gamma_corrected)

        # Return the gamma-mean_std_corrected image
        return normalized

    def mean_std_gamma_preprocessing_4(self, image, names, gamma=0.8):
        # Step 1: Mean-Std Normalization
        image_mean = np.mean(image)
        image_std = np.std(image)
        normalized_image = (image - image_mean) / image_std

        # Step 2: Shift to Positive Values (if necessary)
        shifted_image = normalized_image - np.min(normalized_image)

        # Step 3: Apply Gamma Correction
        gamma_corrected = np.power(shifted_image, gamma)

        # Step 4: Rescale Back to Mean-Std Normalization (Optional)
        normalized = (gamma_corrected - np.mean(gamma_corrected)) / np.std(gamma_corrected)

        # Return the gamma-mean_std_corrected image
        return normalized

    def mean_std_gamma_self(self, image, image_name, sheet_name, gamma_0=0.6, gamma_N=1, gamma_2=1.2):
        """
        Perform mean-std normalization and gamma correction based on the label value from an Excel file.

        Parameters:
        - image: np.ndarray, the input image.
        - excel_file: str, path to the Excel file.
        - sheet_name: str, the sheet name in the Excel file.
        - names_column: str, the column name containing image names.
        - label_column: str, the column name containing labels.
        - image_name: str, the name of the current image to process.
        - gamma_0: float, gamma value for label 0.
        - gamma_N: float, gamma value for label "normal".
        - gamma_2: float, gamma value for label 2.

        Returns:
        - normalized: np.ndarray, the gamma and mean-std corrected image.
        """
        # Step 1: Read the Excel file
        df = pd.read_excel('data/excel/一期单10&二期双10&二期双15-有病理-3456例_20250110.xlsx')

        # Step 2: Find the label for the current image
        label_row = df[df['dcm_name'] == image_name]
        if label_row.empty:
            raise ValueError(f"Image name '{image_name}' not found in the Excel file.")

        label = label_row['dimming'].iloc[0]

        # Step 3: Select the gamma value based on the label
        if label == 0:
            gamma = gamma_0
        elif label == "normal":
            gamma = gamma_N
        elif label == 2:
            gamma = gamma_2
        else:
            raise ValueError(f"Unexpected label value '{label}' for image '{image_name}'.")

        # Step 4: Mean-Std Normalization
        image_mean = np.mean(image)
        image_std = np.std(image)
        normalized_image = (image - image_mean) / image_std

        # Step 5: Shift to Positive Values (if necessary)
        shifted_image = normalized_image - np.min(normalized_image)

        # Step 6: Apply Gamma Correction
        gamma_corrected = np.power(shifted_image, gamma)

        # Step 7: Rescale Back to Mean-Std Normalization (Optional)
        normalized = (gamma_corrected - np.mean(gamma_corrected)) / np.std(gamma_corrected)

        return normalized

    def min_max_normalization_with_clipping(self, image,names, lower_percentile=1, upper_percentile=99):
        lower = np.percentile(image, lower_percentile)
        upper = np.percentile(image, upper_percentile)
        clipped = np.clip(image, lower, upper)
        normalized = (clipped - lower) / (upper - lower)
        return normalized

    def quantile_normalization(self, image, names, reference_distribution=None):
        flat_image = image.flatten()
        sorted_image = np.sort(flat_image)

        if reference_distribution is None:
            reference = sorted_image
        else:
            reference_flat = reference_distribution.flatten()
            reference_sorted = np.sort(reference_flat)
            reference = reference_sorted[:len(sorted_image)]

        # Create a mapping from sorted pixels to reference
        mapping = {pixel: ref for pixel, ref in zip(sorted_image, reference)}
        # Apply the mapping
        normalized_flat = np.array([mapping[pixel] for pixel in flat_image])
        normalized = normalized_flat.reshape(image.shape)
        return normalized

    def standardization_with_scaling(self, image,names, a=0, b=1):
        mean = np.mean(image)
        std = np.std(image)
        standardized = (image - mean) / (std if std != 0 else 1)
        min_val = np.min(standardized)
        max_val = np.max(standardized)
        scaled = a + ((standardized - min_val) / (max_val - min_val)) * (b - a)
        return scaled

    def local_contrast_normalization(self, image, names, size=7, epsilon=1e-8):
        local_mean = uniform_filter(image, size=size)
        local_var = uniform_filter(image ** 2, size=size) - local_mean ** 2
        local_std = np.sqrt(np.maximum(local_var, 0)) + epsilon
        normalized = (image - local_mean) / local_std
        return normalized


    def min_max_normalization(self,  image, names):
        """Normalize using min-max scaling."""
        return (image - np.min(image)) / (np.max(image) - np.min(image))

    def mean_std_normalization(self, image, names):
        """Normalize using mean and standard deviation."""
        return (image - np.mean(image)) / np.std(image)

    def enhance(self,  image):
        """Apply enhancement based on selected method."""
        if self.enhancement_method == 0:
            return image  # 不做增强
        methods = [
            # self.apply_clahe,  # 1: CLAHE
            self.histogram_equalization,  # 2: Histogram Equalization
            lambda x: np.flip(x, axis=1),  # 3: Horizontal Flip
            lambda x: np.flip(x, axis=0),  # 4: Vertical Flip
        ]
        return methods[self.enhancement_method - 1](image)

    def apply_clahe(self, data, clip_limit=2.0, tile_grid_size=(8, 8)):
        """Apply CLAHE to a 2D image slice."""

        # CLAHE 参数
        clip_limit = 2.0
        tile_grid_size = (8, 8)

        # 创建 CLAHE 对象
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        # 存储处理后的切片
        processed_slices = []

        # 对每个时序切片 (第 3 维) 进行 CLAHE 处理
        for i in range(data.shape[2]):
            slice_2d = data[:, :, i]  # 提取第 i 个切片 (102 x 128)
            processed_slice = clahe.apply(slice_2d)  # CLAHE 处理
            processed_slices.append(processed_slice)

        # 合并处理后的切片为三维数组
        processed_data = np.stack(processed_slices, axis=2)

        # clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return  processed_data

    def histogram_equalization(self, image):
        """Apply histogram equalization to a 2D image slice."""
        return cv2.equalizeHist((image * 255).astype(np.uint8)) / 255.0

    #边缘检测
    def edge_detection(self, image):
        """Apply edge detection based on selected method."""
        if self.edge_detection_method == 0:
            return image
        methods = [
            self.sobel_edge_detection,  # 1: Sobel
            self.canny_edge_detection,  # 2: Canny
            self.prewitt_edge_detection,  # 3: Prewitt
            self.roberts_edge_detection,  # 4: Roberts
            self.log_edge_detection,  # 5: Laplacian of Gaussian
            self.dog_edge_detection,  # 6: Difference of Gaussians
            self.scharr_edge_detection,  # 7: Scharr
            self.gabor_edge_detection,  # 8: Gabor
            self.hessian_edge_detection,  # 9: Hessian
            self.morphological_edge_detection,  # 10: Morphological
            self.wavelet_edge_detection  # 11: Wavelet Transform
        ]
        return methods[self.edge_detection_method - 1](image)

    def sobel_edge_detection(self, image):
        """Sobel Edge Detection."""
        return filters.sobel(image)

    def canny_edge_detection(self, image):
        """Canny Edge Detection."""
        return canny(image)

    def prewitt_edge_detection(self, image):
        """Prewitt Edge Detection."""
        return filters.prewitt(image)

    def roberts_edge_detection(self, image):
        """Roberts Edge Detection."""
        return roberts(image)

    def log_edge_detection(self, image, sigma=2):
        """Laplacian of Gaussian (LoG) Edge Detection."""
        return gaussian_laplace(image, sigma=sigma)

    def dog_edge_detection(self, image, low_sigma=1, high_sigma=2):
        """Difference of Gaussians (DoG) Edge Detection."""
        return difference_of_gaussians(image, low_sigma=low_sigma, high_sigma=high_sigma)

    def scharr_edge_detection(self, image):
        """Scharr Edge Detection."""
        return filters.scharr(image)

    def gabor_edge_detection(self, image, frequency=0.6):
        """Gabor Filter (Edge Detection by Directional Enhancement)."""
        gabor_real, _ = gabor(image, frequency=frequency)
        return gabor_real

    def hessian_edge_detection(self, image, sigma=3):
        """Hessian-based Edge Detection."""
        # Using skimage's hessian_matrix function to compute the Hessian
        hessian_res = filters.hessian(image)
        return np.sqrt(hessian_res[0] ** 2 + hessian_res[1] ** 2)  # Magnitude of the Hessian matrix

    def morphological_edge_detection(self, image):
        """Morphological Edge Detection."""
        # Using dilation and erosion to detect edges
        dilated_image = morphology.dilation(image)
        eroded_image = morphology.erosion(image)
        return dilated_image - eroded_image  # Edge is the difference between dilation and erosion

    def wavelet_edge_detection(self, image, wavelet='haar'):
        """Wavelet Transform for Edge Detection using PyWavelets."""
        # Perform 2D discrete wavelet transform
        coeffs2 = pywt.dwt2(image, wavelet)
        LL, (LH, HL, HH) = coeffs2
        # Combine the horizontal, vertical, and diagonal details (LH, HL, HH)
        edge_image = np.sqrt(LH**2 + HL**2 + HH**2)
        return edge_image