import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse

def calculate_ssim(imageA, imageB):
    return ssim(imageA, imageB, data_range=1)

def calculate_snr(imageA, imageB):
    noise = imageA - imageB
    signal_power = np.mean(imageA ** 2)
    noise_power = np.mean(noise ** 2)
    return 10 * np.log10(signal_power / noise_power)

def calculate_psnr(imageA, imageB):
    return psnr(imageA, imageB)

def calculate_mse(imageA, imageB):
    return mse(imageA, imageB)

def calculate_detailed_variance(image, threshold=0.1):
    """
    Calculate the variance of the detailed part of the image.
    Pixels with values greater than the threshold are considered as detailed part.
    """
    detailed_part = image[image > threshold]
    return np.var(detailed_part)

def calculate_background_variance(image, threshold=0.1):
    """
    Calculate the variance of the background part of the image.
    Pixels with values less than or equal to the threshold are considered as background part.
    """
    background_part = image[image <= threshold]
    return np.var(background_part)

# サンプルデータ
imageA = np.random.rand(256, 256)
imageB = imageA + np.random.normal(scale=0.1, size=imageA.shape)

# 計算結果の表示
print("SSIM:", calculate_ssim(imageA, imageB))
print("SNR:", calculate_snr(imageA, imageB))
print("PSNR:", calculate_psnr(imageA, imageB))
print("MSE:", calculate_mse(imageA, imageB))

# 詳細分散と背景分散の計算結果の表示
threshold = 0.1
print("Detailed Variance:", calculate_detailed_variance(imageA, threshold))
print("Background Variance:", calculate_background_variance(imageA, threshold))
