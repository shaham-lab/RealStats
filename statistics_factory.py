# statistics_factory.py

from typing import Optional

from processing.blurness_histogram import BlurDetectionHistogram
from processing.filters_histograms import DirectionalEdgeNormHistogram, EdgeNormHistogram, EmbossNormHistogram, GaussianDifferenceNormHistogram, HighPassNormHistogram, NoiseNormHistogram, SharpnessNormHistogram, SmoothingNormHistogram
from processing.gabor_histogram import GaborFilterFeatureMapNorm
from processing.histograms import DCTNormHistogram, FourierNormHistogram, WaveletNormHistogram
from processing.hsv_histogram import HSVHistogram
from processing.jpeg_histogram import JPEGCompressionRatioHistogram
from processing.laplacian_histogram import LaplacianVarianceHistogram
from processing.manifold_bias_histogram import LatentNoiseCriterion
from processing.psnr_noise_histogram import PSNRBlurHistogram
from processing.rigid_histogram import RIGIDBEiTHistogram, RIGIDBigGCLIPHistogram, RIGIDConvNeXtHistogram, RIGIDDinoV2Histogram, RIGIDEVAHistogram, RIGIDOpenAICLIPHistogram, RIGIDOpenCLIPHistogram, RIGIDResNet50Histogram
from processing.sift_histogram import SIFTHistogram
from processing.ssim_noise_histogram import SSIMBlurHistogram

# Import all histogram classes


# Define a dictionary that maps statistic names to their corresponding histogram generator classes
STATISTIC_HISTOGRAMS = {
    # Wavelet and Frequency Domain Statistics
    'bior1.1': lambda level: WaveletNormHistogram(selected_indices=[level], wave='bior1.1'),
    'bior3.1': lambda level: WaveletNormHistogram(selected_indices=[level], wave='bior3.1'),
    'bior6.8': lambda level: WaveletNormHistogram(selected_indices=[level], wave='bior6.8'),
    'coif1': lambda level: WaveletNormHistogram(selected_indices=[level], wave='coif1'),
    'coif10': lambda level: WaveletNormHistogram(selected_indices=[level], wave='coif10'),
    'db1': lambda level: WaveletNormHistogram(selected_indices=[level], wave='db1'),
    'db38': lambda level: WaveletNormHistogram(selected_indices=[level], wave='db38'),
    'haar': lambda level: WaveletNormHistogram(selected_indices=[level], wave='haar'),
    'rbio6.8': lambda level: WaveletNormHistogram(selected_indices=[level], wave='rbio6.8'),
    'sym2': lambda level: WaveletNormHistogram(selected_indices=[level], wave='sym2'),

    'fourier': lambda level: FourierNormHistogram() if level == 0 else None,
    'dct': lambda level: DCTNormHistogram(dct_type=level) if 0 < level <= 4 else None,

    # Non-Wavelet Image Statistics
    'blurness': lambda level: BlurDetectionHistogram() if level == 0 else None,
    'gabor': lambda level: GaborFilterFeatureMapNorm() if level == 0 else None,
    'hsv': lambda level: HSVHistogram() if level == 0 else None,
    'jpeg': lambda level: JPEGCompressionRatioHistogram() if level == 0 else None,
    'laplacian': lambda level: LaplacianVarianceHistogram() if level == 0 else None,
    'psnr': lambda level: PSNRBlurHistogram() if level == 0 else None,
    'sift': lambda level: SIFTHistogram() if level == 0 else None,
    'ssim': lambda level: SSIMBlurHistogram() if level == 0 else None,

    # New Latent Space Statistics
    'edge5x5': lambda level: EdgeNormHistogram() if level == 0 else None,
    'smoothing': lambda level: SmoothingNormHistogram() if level == 0 else None,
    'noise': lambda level: NoiseNormHistogram() if level == 0 else None,
    'sharpness': lambda level: SharpnessNormHistogram() if level == 0 else None,
    'emboss': lambda level: EmbossNormHistogram() if level == 0 else None,
    'highpass': lambda level: HighPassNormHistogram() if level == 0 else None,
    'sobel': lambda level: DirectionalEdgeNormHistogram() if level == 0 else None,
    'gauss_diff': lambda level: GaussianDifferenceNormHistogram() if level == 0 else None,

    # Paper Statistics
    'LatentNoiseCriterion': lambda level: LatentNoiseCriterion() if level == 0 else None,

    # ==============================
    #    Hardcoded RIGID Statistics
    # ==============================
    
    # ===== DINO =====
    'RIGID.DINO.001': lambda level: RIGIDDinoV2Histogram(noise_level=0.001) if level == 0 else None,
    'RIGID.DINO.01':  lambda level: RIGIDDinoV2Histogram(noise_level=0.01)  if level == 0 else None,
    'RIGID.DINO.05':  lambda level: RIGIDDinoV2Histogram(noise_level=0.05)  if level == 0 else None,
    'RIGID.DINO.10':  lambda level: RIGIDDinoV2Histogram(noise_level=0.10)  if level == 0 else None,
    'RIGID.DINO.20':  lambda level: RIGIDDinoV2Histogram(noise_level=0.20)  if level == 0 else None,
    'RIGID.DINO.30':  lambda level: RIGIDDinoV2Histogram(noise_level=0.30)  if level == 0 else None,
    'RIGID.DINO.50':  lambda level: RIGIDDinoV2Histogram(noise_level=0.50)  if level == 0 else None,
    'RIGID.DINO.75':  lambda level: RIGIDDinoV2Histogram(noise_level=0.75)  if level == 0 else None,
    'RIGID.DINO.100': lambda level: RIGIDDinoV2Histogram(noise_level=1.0)   if level == 0 else None,

    # ===== DINOv3 =====
    # --- ViT-7B/16 ---
    'RIGID.DINOV3.VIT7B16.001': lambda level: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vit7b16-pretrain-lvd1689m", noise_level=0.001) if level == 0 else None,
    'RIGID.DINOV3.VIT7B16.01':  lambda level: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vit7b16-pretrain-lvd1689m", noise_level=0.01)  if level == 0 else None,
    'RIGID.DINOV3.VIT7B16.05':  lambda level: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vit7b16-pretrain-lvd1689m", noise_level=0.05)  if level == 0 else None,
    'RIGID.DINOV3.VIT7B16.10':  lambda level: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vit7b16-pretrain-lvd1689m", noise_level=0.10)  if level == 0 else None,
    'RIGID.DINOV3.VIT7B16.20':  lambda level: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vit7b16-pretrain-lvd1689m", noise_level=0.20)  if level == 0 else None,
    'RIGID.DINOV3.VIT7B16.30':  lambda level: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vit7b16-pretrain-lvd1689m", noise_level=0.30)  if level == 0 else None,
    'RIGID.DINOV3.VIT7B16.50':  lambda level: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vit7b16-pretrain-lvd1689m", noise_level=0.50)  if level == 0 else None,
    'RIGID.DINOV3.VIT7B16.75':  lambda level: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vit7b16-pretrain-lvd1689m", noise_level=0.75)  if level == 0 else None,
    'RIGID.DINOV3.VIT7B16.100': lambda level: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vit7b16-pretrain-lvd1689m", noise_level=1.0)   if level == 0 else None,

    # --- ViT-S/16 ---
    'RIGID.DINOV3.VITS16.001': lambda level: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vits16-pretrain-lvd1689m", noise_level=0.001) if level == 0 else None,
    'RIGID.DINOV3.VITS16.01':  lambda level: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vits16-pretrain-lvd1689m", noise_level=0.01)  if level == 0 else None,
    'RIGID.DINOV3.VITS16.05':  lambda level: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vits16-pretrain-lvd1689m", noise_level=0.05)  if level == 0 else None,
    'RIGID.DINOV3.VITS16.10':  lambda level: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vits16-pretrain-lvd1689m", noise_level=0.10)  if level == 0 else None,
    'RIGID.DINOV3.VITS16.20':  lambda level: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vits16-pretrain-lvd1689m", noise_level=0.20)  if level == 0 else None,
    'RIGID.DINOV3.VITS16.30':  lambda level: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vits16-pretrain-lvd1689m", noise_level=0.30)  if level == 0 else None,
    'RIGID.DINOV3.VITS16.50':  lambda level: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vits16-pretrain-lvd1689m", noise_level=0.50)  if level == 0 else None,
    'RIGID.DINOV3.VITS16.75':  lambda level: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vits16-pretrain-lvd1689m", noise_level=0.75)  if level == 0 else None,
    'RIGID.DINOV3.VITS16.100': lambda level: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vits16-pretrain-lvd1689m", noise_level=1.0)   if level == 0 else None,

    # --- ConvNeXt-Small ---
    'RIGID.DINOV3.CONVNEXTSMALL.001': lambda level: RIGIDDinoV3Histogram(model_name="facebook/dinov3-convnext-small-pretrain-lvd1689m", noise_level=0.001) if level == 0 else None,
    'RIGID.DINOV3.CONVNEXTSMALL.01':  lambda level: RIGIDDinoV3Histogram(model_name="facebook/dinov3-convnext-small-pretrain-lvd1689m", noise_level=0.01)  if level == 0 else None,
    'RIGID.DINOV3.CONVNEXTSMALL.05':  lambda level: RIGIDDinoV3Histogram(model_name="facebook/dinov3-convnext-small-pretrain-lvd1689m", noise_level=0.05)  if level == 0 else None,
    'RIGID.DINOV3.CONVNEXTSMALL.10':  lambda level: RIGIDDinoV3Histogram(model_name="facebook/dinov3-convnext-small-pretrain-lvd1689m", noise_level=0.10)  if level == 0 else None,
    'RIGID.DINOV3.CONVNEXTSMALL.20':  lambda level: RIGIDDinoV3Histogram(model_name="facebook/dinov3-convnext-small-pretrain-lvd1689m", noise_level=0.20)  if level == 0 else None,
    'RIGID.DINOV3.CONVNEXTSMALL.30':  lambda level: RIGIDDinoV3Histogram(model_name="facebook/dinov3-convnext-small-pretrain-lvd1689m", noise_level=0.30)  if level == 0 else None,
    'RIGID.DINOV3.CONVNEXTSMALL.50':  lambda level: RIGIDDinoV3Histogram(model_name="facebook/dinov3-convnext-small-pretrain-lvd1689m", noise_level=0.50)  if level == 0 else None,
    'RIGID.DINOV3.CONVNEXTSMALL.75':  lambda level: RIGIDDinoV3Histogram(model_name="facebook/dinov3-convnext-small-pretrain-lvd1689m", noise_level=0.75)  if level == 0 else None,
    'RIGID.DINOV3.CONVNEXTSMALL.100': lambda level: RIGIDDinoV3Histogram(model_name="facebook/dinov3-convnext-small-pretrain-lvd1689m", noise_level=1.0)   if level == 0 else None,

    # ===== EVA =====
    'RIGID.EVA.001': lambda level: RIGIDEVAHistogram(noise_level=0.001) if level == 0 else None,
    'RIGID.EVA.01':  lambda level: RIGIDEVAHistogram(noise_level=0.01)  if level == 0 else None,
    'RIGID.EVA.05':  lambda level: RIGIDEVAHistogram(noise_level=0.05)  if level == 0 else None,
    'RIGID.EVA.10':  lambda level: RIGIDEVAHistogram(noise_level=0.10)  if level == 0 else None,
    'RIGID.EVA.20':  lambda level: RIGIDEVAHistogram(noise_level=0.20)  if level == 0 else None,
    'RIGID.EVA.30':  lambda level: RIGIDEVAHistogram(noise_level=0.30)  if level == 0 else None,
    'RIGID.EVA.50':  lambda level: RIGIDEVAHistogram(noise_level=0.50)  if level == 0 else None,
    'RIGID.EVA.75':  lambda level: RIGIDEVAHistogram(noise_level=0.75)  if level == 0 else None,
    'RIGID.EVA.100': lambda level: RIGIDEVAHistogram(noise_level=1.0)   if level == 0 else None,

    # ===== BEiT =====
    'RIGID.BEIT.001': lambda level: RIGIDBEiTHistogram(noise_level=0.001) if level == 0 else None,
    'RIGID.BEIT.01':  lambda level: RIGIDBEiTHistogram(noise_level=0.01)  if level == 0 else None,
    'RIGID.BEIT.05':  lambda level: RIGIDBEiTHistogram(noise_level=0.05)  if level == 0 else None,
    'RIGID.BEIT.10':  lambda level: RIGIDBEiTHistogram(noise_level=0.10)  if level == 0 else None,
    'RIGID.BEIT.20':  lambda level: RIGIDBEiTHistogram(noise_level=0.20)  if level == 0 else None,
    'RIGID.BEIT.30':  lambda level: RIGIDBEiTHistogram(noise_level=0.30)  if level == 0 else None,
    'RIGID.BEIT.50':  lambda level: RIGIDBEiTHistogram(noise_level=0.50)  if level == 0 else None,
    'RIGID.BEIT.75':  lambda level: RIGIDBEiTHistogram(noise_level=0.75)  if level == 0 else None,
    'RIGID.BEIT.100': lambda level: RIGIDBEiTHistogram(noise_level=1.0)   if level == 0 else None,

    # ===== OpenCLIP ViT-H =====
    'RIGID.CLIP.001': lambda level: RIGIDOpenCLIPHistogram(noise_level=0.001) if level == 0 else None,
    'RIGID.CLIP.01':  lambda level: RIGIDOpenCLIPHistogram(noise_level=0.01)  if level == 0 else None,
    'RIGID.CLIP.05':  lambda level: RIGIDOpenCLIPHistogram(noise_level=0.05)  if level == 0 else None,
    'RIGID.CLIP.10':  lambda level: RIGIDOpenCLIPHistogram(noise_level=0.10)  if level == 0 else None,
    'RIGID.CLIP.20':  lambda level: RIGIDOpenCLIPHistogram(noise_level=0.20)  if level == 0 else None,
    'RIGID.CLIP.30':  lambda level: RIGIDOpenCLIPHistogram(noise_level=0.30)  if level == 0 else None,
    'RIGID.CLIP.50':  lambda level: RIGIDOpenCLIPHistogram(noise_level=0.50)  if level == 0 else None,
    'RIGID.CLIP.75':  lambda level: RIGIDOpenCLIPHistogram(noise_level=0.75)  if level == 0 else None,
    'RIGID.CLIP.100': lambda level: RIGIDOpenCLIPHistogram(noise_level=1.0)   if level == 0 else None,

    # ===== OpenAI CLIP ViT-L/14 =====
    'RIGID.CLIPOPENAI.001': lambda level: RIGIDOpenAICLIPHistogram(noise_level=0.001) if level == 0 else None,
    'RIGID.CLIPOPENAI.01':  lambda level: RIGIDOpenAICLIPHistogram(noise_level=0.01)  if level == 0 else None,
    'RIGID.CLIPOPENAI.05':  lambda level: RIGIDOpenAICLIPHistogram(noise_level=0.05)  if level == 0 else None,
    'RIGID.CLIPOPENAI.10':  lambda level: RIGIDOpenAICLIPHistogram(noise_level=0.10)  if level == 0 else None,
    'RIGID.CLIPOPENAI.20':  lambda level: RIGIDOpenAICLIPHistogram(noise_level=0.20)  if level == 0 else None,
    'RIGID.CLIPOPENAI.30':  lambda level: RIGIDOpenAICLIPHistogram(noise_level=0.30)  if level == 0 else None,
    'RIGID.CLIPOPENAI.50':  lambda level: RIGIDOpenAICLIPHistogram(noise_level=0.50)  if level == 0 else None,
    'RIGID.CLIPOPENAI.75':  lambda level: RIGIDOpenAICLIPHistogram(noise_level=0.75)  if level == 0 else None,
    'RIGID.CLIPOPENAI.100': lambda level: RIGIDOpenAICLIPHistogram(noise_level=1.0)   if level == 0 else None,

    # ===== LAION CLIP ViT-bigG-14 =====
    'RIGID.CLIPBIGG.001': lambda level: RIGIDBigGCLIPHistogram(noise_level=0.001) if level == 0 else None,
    'RIGID.CLIPBIGG.01':  lambda level: RIGIDBigGCLIPHistogram(noise_level=0.01)  if level == 0 else None,
    'RIGID.CLIPBIGG.05':  lambda level: RIGIDBigGCLIPHistogram(noise_level=0.05)  if level == 0 else None,
    'RIGID.CLIPBIGG.10':  lambda level: RIGIDBigGCLIPHistogram(noise_level=0.10)  if level == 0 else None,
    'RIGID.CLIPBIGG.20':  lambda level: RIGIDBigGCLIPHistogram(noise_level=0.20)  if level == 0 else None,
    'RIGID.CLIPBIGG.30':  lambda level: RIGIDBigGCLIPHistogram(noise_level=0.30)  if level == 0 else None,
    'RIGID.CLIPBIGG.50':  lambda level: RIGIDBigGCLIPHistogram(noise_level=0.50)  if level == 0 else None,
    'RIGID.CLIPBIGG.75':  lambda level: RIGIDBigGCLIPHistogram(noise_level=0.75)  if level == 0 else None,
    'RIGID.CLIPBIGG.100': lambda level: RIGIDBigGCLIPHistogram(noise_level=1.0)   if level == 0 else None,

    # ===== ConvNeXt =====
    'RIGID.CONVNEXT.001': lambda level: RIGIDConvNeXtHistogram(noise_level=0.001) if level == 0 else None,
    'RIGID.CONVNEXT.01':  lambda level: RIGIDConvNeXtHistogram(noise_level=0.01)  if level == 0 else None,
    'RIGID.CONVNEXT.05':  lambda level: RIGIDConvNeXtHistogram(noise_level=0.05)  if level == 0 else None,
    'RIGID.CONVNEXT.10':  lambda level: RIGIDConvNeXtHistogram(noise_level=0.10)  if level == 0 else None,
    'RIGID.CONVNEXT.20':  lambda level: RIGIDConvNeXtHistogram(noise_level=0.20)  if level == 0 else None,
    'RIGID.CONVNEXT.30':  lambda level: RIGIDConvNeXtHistogram(noise_level=0.30)  if level == 0 else None,
    'RIGID.CONVNEXT.50':  lambda level: RIGIDConvNeXtHistogram(noise_level=0.50)  if level == 0 else None,
    'RIGID.CONVNEXT.75':  lambda level: RIGIDConvNeXtHistogram(noise_level=0.75)  if level == 0 else None,
    'RIGID.CONVNEXT.100': lambda level: RIGIDConvNeXtHistogram(noise_level=1.0)   if level == 0 else None,

    # ===== ResNet-50 =====
    'RIGID.RESNET.001': lambda level: RIGIDResNet50Histogram(noise_level=0.001) if level == 0 else None,
    'RIGID.RESNET.01':  lambda level: RIGIDResNet50Histogram(noise_level=0.01)  if level == 0 else None,
    'RIGID.RESNET.05':  lambda level: RIGIDResNet50Histogram(noise_level=0.05)  if level == 0 else None,
    'RIGID.RESNET.10':  lambda level: RIGIDResNet50Histogram(noise_level=0.10)  if level == 0 else None,
    'RIGID.RESNET.20':  lambda level: RIGIDResNet50Histogram(noise_level=0.20)  if level == 0 else None,
    'RIGID.RESNET.30':  lambda level: RIGIDResNet50Histogram(noise_level=0.30)  if level == 0 else None,
    'RIGID.RESNET.50':  lambda level: RIGIDResNet50Histogram(noise_level=0.50)  if level == 0 else None,
    'RIGID.RESNET.75':  lambda level: RIGIDResNet50Histogram(noise_level=0.75)  if level == 0 else None,
    'RIGID.RESNET.100': lambda level: RIGIDResNet50Histogram(noise_level=1.0)   if level == 0 else None,
}


def get_histogram_generator(statistic: str, level: int) -> Optional[object]:
    """Returns the appropriate histogram generator based on statistic name and level."""
    return STATISTIC_HISTOGRAMS.get(statistic, lambda lvl: None)(level)
