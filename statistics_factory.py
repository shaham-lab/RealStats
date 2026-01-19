# statistics_factory.py

from typing import Optional

from processing.blurness_histogram import BlurDetectionHistogram
from processing.filters_histograms import DirectionalEdgeNormHistogram, EdgeNormHistogram, EmbossNormHistogram, GaussianDifferenceNormHistogram, HighPassNormHistogram, NoiseNormHistogram, SharpnessNormHistogram, SmoothingNormHistogram
from processing.gabor_histogram import GaborFilterFeatureMapNorm
from processing.waves import WaveletNormHistogram
from processing.hsv_histogram import HSVHistogram
from processing.jpeg_histogram import JPEGCompressionRatioHistogram
from processing.laplacian_histogram import LaplacianVarianceHistogram
from processing.manifold_bias_histogram import LatentNoiseCriterion, LatentNoiseCriterionOriginal
from processing.psnr_noise_histogram import PSNRBlurHistogram
from processing.rigid_histogram import RIGIDBEiTHistogram, RIGIDBigGCLIPHistogram, RIGIDConvNeXtHistogram, RIGIDDinoV2Histogram, RIGIDDinoV3Histogram, RIGIDEVAHistogram, RIGIDOpenAICLIPHistogram, RIGIDOpenCLIPHistogram, RIGIDResNet50Histogram
from processing.sift_histogram import SIFTHistogram
from processing.ssim_noise_histogram import SSIMBlurHistogram
from processing.waves import DCTNormHistogram, FourierNormHistogram


# Define a dictionary that maps statistic names to their corresponding histogram generator classes
STATISTIC_HISTOGRAMS = {
    # Wavelet and Frequency Domain Statistics
    'bior1.1_l0': lambda: WaveletNormHistogram(selected_indices=[0], wave='bior1.1'),
    'bior3.1_l0': lambda: WaveletNormHistogram(selected_indices=[0], wave='bior3.1'),
    'bior6.8_l0': lambda: WaveletNormHistogram(selected_indices=[0], wave='bior6.8'),
    'coif1_l0': lambda: WaveletNormHistogram(selected_indices=[0], wave='coif1'),
    'coif10_l0': lambda: WaveletNormHistogram(selected_indices=[0], wave='coif10'),
    'db1_l0': lambda: WaveletNormHistogram(selected_indices=[0], wave='db1'),
    'db38_l0': lambda: WaveletNormHistogram(selected_indices=[0], wave='db38'),
    'haar_l0': lambda: WaveletNormHistogram(selected_indices=[0], wave='haar'),
    'rbio6.8_l0': lambda: WaveletNormHistogram(selected_indices=[0], wave='rbio6.8'),
    'sym2_l0': lambda: WaveletNormHistogram(selected_indices=[0], wave='sym2'),

    'fourier': lambda: FourierNormHistogram(),
    'dct_l1': lambda: DCTNormHistogram(dct_type=1),

    # Non-Wavelet Image Statistics
    'blurness': lambda: BlurDetectionHistogram(),
    'gabor': lambda: GaborFilterFeatureMapNorm(),
    'hsv': lambda: HSVHistogram(),
    'jpeg': lambda: JPEGCompressionRatioHistogram(),
    'laplacian': lambda: LaplacianVarianceHistogram(),
    'psnr': lambda: PSNRBlurHistogram(),
    'sift': lambda: SIFTHistogram(),
    'ssim': lambda: SSIMBlurHistogram(),

    # New Latent Space Statistics
    'edge5x5': lambda: EdgeNormHistogram(),
    'smoothing': lambda: SmoothingNormHistogram(),
    'noise': lambda: NoiseNormHistogram(),
    'sharpness': lambda: SharpnessNormHistogram(),
    'emboss': lambda: EmbossNormHistogram(),
    'highpass': lambda: HighPassNormHistogram(),
    'sobel': lambda: DirectionalEdgeNormHistogram(),
    'gauss_diff': lambda: GaussianDifferenceNormHistogram(),

    # Paper Statistics
    'LatentNoiseCriterion': lambda: LatentNoiseCriterion(),
    'LatentNoiseCriterion_original': lambda: LatentNoiseCriterionOriginal(scores_csv="manifold_bias_criterion.csv"),

    # ==============================
    #    Hardcoded RIGID Statistics
    # ==============================
    
    # ===== DINO =====
    'RIGID.DINO.001': lambda: RIGIDDinoV2Histogram(noise_level=0.001),
    'RIGID.DINO.01': lambda: RIGIDDinoV2Histogram(noise_level=0.01),
    'RIGID.DINO.05': lambda: RIGIDDinoV2Histogram(noise_level=0.05),
    'RIGID.DINO.10': lambda: RIGIDDinoV2Histogram(noise_level=0.10),
    'RIGID.DINO.20': lambda: RIGIDDinoV2Histogram(noise_level=0.20),
    'RIGID.DINO.30': lambda: RIGIDDinoV2Histogram(noise_level=0.30),
    'RIGID.DINO.50': lambda: RIGIDDinoV2Histogram(noise_level=0.50),
    'RIGID.DINO.75': lambda: RIGIDDinoV2Histogram(noise_level=0.75),
    'RIGID.DINO.100': lambda: RIGIDDinoV2Histogram(noise_level=1.0),

    # ===== DINOv3 =====
    # TODO: Not feasible.
    # # --- ViT-7B/16 ---
    # 'RIGID.DINOV3.VIT7B16.001': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vit7b16-pretrain-lvd1689m", noise_level=0.001),
    # 'RIGID.DINOV3.VIT7B16.01': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vit7b16-pretrain-lvd1689m", noise_level=0.01),
    # 'RIGID.DINOV3.VIT7B16.05': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vit7b16-pretrain-lvd1689m", noise_level=0.05),
    # 'RIGID.DINOV3.VIT7B16.10': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vit7b16-pretrain-lvd1689m", noise_level=0.10),
    # 'RIGID.DINOV3.VIT7B16.20': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vit7b16-pretrain-lvd1689m", noise_level=0.20),
    # 'RIGID.DINOV3.VIT7B16.30': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vit7b16-pretrain-lvd1689m", noise_level=0.30),
    # 'RIGID.DINOV3.VIT7B16.50': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vit7b16-pretrain-lvd1689m", noise_level=0.50),
    # 'RIGID.DINOV3.VIT7B16.75': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vit7b16-pretrain-lvd1689m", noise_level=0.75),
    # 'RIGID.DINOV3.VIT7B16.100': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vit7b16-pretrain-lvd1689m", noise_level=1.0),

    # --- ViT-H/16 ---
    'RIGID.DINOV3.VITH16.001': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vith16plus-pretrain-lvd1689m", noise_level=0.001),
    'RIGID.DINOV3.VITH16.01': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vith16plus-pretrain-lvd1689m", noise_level=0.01),
    'RIGID.DINOV3.VITH16.05': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vith16plus-pretrain-lvd1689m", noise_level=0.05),
    'RIGID.DINOV3.VITH16.10': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vith16plus-pretrain-lvd1689m", noise_level=0.10),
    'RIGID.DINOV3.VITH16.20': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vith16plus-pretrain-lvd1689m", noise_level=0.20),
    'RIGID.DINOV3.VITH16.30': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vith16plus-pretrain-lvd1689m", noise_level=0.30),
    'RIGID.DINOV3.VITH16.50': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vith16plus-pretrain-lvd1689m", noise_level=0.50),
    'RIGID.DINOV3.VITH16.75': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vith16plus-pretrain-lvd1689m", noise_level=0.75),
    'RIGID.DINOV3.VITH16.100': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vith16plus-pretrain-lvd1689m", noise_level=1.0),

    # --- ViT-S/16 ---
    'RIGID.DINOV3.VITS16.001': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vits16-pretrain-lvd1689m", noise_level=0.001),
    'RIGID.DINOV3.VITS16.01': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vits16-pretrain-lvd1689m", noise_level=0.01),
    'RIGID.DINOV3.VITS16.05': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vits16-pretrain-lvd1689m", noise_level=0.05),
    'RIGID.DINOV3.VITS16.10': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vits16-pretrain-lvd1689m", noise_level=0.10),
    'RIGID.DINOV3.VITS16.20': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vits16-pretrain-lvd1689m", noise_level=0.20),
    'RIGID.DINOV3.VITS16.30': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vits16-pretrain-lvd1689m", noise_level=0.30),
    'RIGID.DINOV3.VITS16.50': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vits16-pretrain-lvd1689m", noise_level=0.50),
    'RIGID.DINOV3.VITS16.75': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vits16-pretrain-lvd1689m", noise_level=0.75),
    'RIGID.DINOV3.VITS16.100': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-vits16-pretrain-lvd1689m", noise_level=1.0),

    # --- ConvNeXt-Small ---
    'RIGID.DINOV3.CONVNEXTSMALL.001': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-convnext-small-pretrain-lvd1689m", noise_level=0.001),
    'RIGID.DINOV3.CONVNEXTSMALL.01': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-convnext-small-pretrain-lvd1689m", noise_level=0.01),
    'RIGID.DINOV3.CONVNEXTSMALL.05': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-convnext-small-pretrain-lvd1689m", noise_level=0.05),
    'RIGID.DINOV3.CONVNEXTSMALL.10': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-convnext-small-pretrain-lvd1689m", noise_level=0.10),
    'RIGID.DINOV3.CONVNEXTSMALL.20': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-convnext-small-pretrain-lvd1689m", noise_level=0.20),
    'RIGID.DINOV3.CONVNEXTSMALL.30': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-convnext-small-pretrain-lvd1689m", noise_level=0.30),
    'RIGID.DINOV3.CONVNEXTSMALL.50': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-convnext-small-pretrain-lvd1689m", noise_level=0.50),
    'RIGID.DINOV3.CONVNEXTSMALL.75': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-convnext-small-pretrain-lvd1689m", noise_level=0.75),
    'RIGID.DINOV3.CONVNEXTSMALL.100': lambda: RIGIDDinoV3Histogram(model_name="facebook/dinov3-convnext-small-pretrain-lvd1689m", noise_level=1.0),

    # ===== BEiT =====
    'RIGID.BEIT.001': lambda: RIGIDBEiTHistogram(noise_level=0.001),
    'RIGID.BEIT.01': lambda: RIGIDBEiTHistogram(noise_level=0.01),
    'RIGID.BEIT.05': lambda: RIGIDBEiTHistogram(noise_level=0.05),
    'RIGID.BEIT.10': lambda: RIGIDBEiTHistogram(noise_level=0.10),
    'RIGID.BEIT.20': lambda: RIGIDBEiTHistogram(noise_level=0.20),
    'RIGID.BEIT.30': lambda: RIGIDBEiTHistogram(noise_level=0.30),
    'RIGID.BEIT.50': lambda: RIGIDBEiTHistogram(noise_level=0.50),
    'RIGID.BEIT.75': lambda: RIGIDBEiTHistogram(noise_level=0.75),
    'RIGID.BEIT.100': lambda: RIGIDBEiTHistogram(noise_level=1.0),

    # ===== OpenCLIP ViT-H =====
    'RIGID.CLIP.001': lambda: RIGIDOpenCLIPHistogram(noise_level=0.001),
    'RIGID.CLIP.01': lambda: RIGIDOpenCLIPHistogram(noise_level=0.01),
    'RIGID.CLIP.05': lambda: RIGIDOpenCLIPHistogram(noise_level=0.05),
    'RIGID.CLIP.10': lambda: RIGIDOpenCLIPHistogram(noise_level=0.10),
    'RIGID.CLIP.20': lambda: RIGIDOpenCLIPHistogram(noise_level=0.20),
    'RIGID.CLIP.30': lambda: RIGIDOpenCLIPHistogram(noise_level=0.30),
    'RIGID.CLIP.50': lambda: RIGIDOpenCLIPHistogram(noise_level=0.50),
    'RIGID.CLIP.75': lambda: RIGIDOpenCLIPHistogram(noise_level=0.75),
    'RIGID.CLIP.100': lambda: RIGIDOpenCLIPHistogram(noise_level=1.0),

    # ===== OpenAI CLIP ViT-L/14 =====
    'RIGID.CLIPOPENAI.001': lambda: RIGIDOpenAICLIPHistogram(noise_level=0.001),
    'RIGID.CLIPOPENAI.01': lambda: RIGIDOpenAICLIPHistogram(noise_level=0.01),
    'RIGID.CLIPOPENAI.05': lambda: RIGIDOpenAICLIPHistogram(noise_level=0.05),
    'RIGID.CLIPOPENAI.10': lambda: RIGIDOpenAICLIPHistogram(noise_level=0.10),
    'RIGID.CLIPOPENAI.20': lambda: RIGIDOpenAICLIPHistogram(noise_level=0.20),
    'RIGID.CLIPOPENAI.30': lambda: RIGIDOpenAICLIPHistogram(noise_level=0.30),
    'RIGID.CLIPOPENAI.50': lambda: RIGIDOpenAICLIPHistogram(noise_level=0.50),
    'RIGID.CLIPOPENAI.75': lambda: RIGIDOpenAICLIPHistogram(noise_level=0.75),
    'RIGID.CLIPOPENAI.100': lambda: RIGIDOpenAICLIPHistogram(noise_level=1.0),

    # # ===== LAION CLIP ViT-bigG-14 =====
    # TODO: too slow.
    # 'RIGID.CLIPBIGG.001': lambda: RIGIDBigGCLIPHistogram(noise_level=0.001),
    # 'RIGID.CLIPBIGG.01': lambda: RIGIDBigGCLIPHistogram(noise_level=0.01),
    # 'RIGID.CLIPBIGG.05': lambda: RIGIDBigGCLIPHistogram(noise_level=0.05),
    # 'RIGID.CLIPBIGG.10': lambda: RIGIDBigGCLIPHistogram(noise_level=0.10),
    # 'RIGID.CLIPBIGG.20': lambda: RIGIDBigGCLIPHistogram(noise_level=0.20),
    # 'RIGID.CLIPBIGG.30': lambda: RIGIDBigGCLIPHistogram(noise_level=0.30),
    # 'RIGID.CLIPBIGG.50': lambda: RIGIDBigGCLIPHistogram(noise_level=0.50),
    # 'RIGID.CLIPBIGG.75': lambda: RIGIDBigGCLIPHistogram(noise_level=0.75),
    # 'RIGID.CLIPBIGG.100': lambda: RIGIDBigGCLIPHistogram(noise_level=1.0),

    # ===== ConvNeXt =====
    'RIGID.CONVNEXT.001': lambda: RIGIDConvNeXtHistogram(noise_level=0.001),
    'RIGID.CONVNEXT.01': lambda: RIGIDConvNeXtHistogram(noise_level=0.01),
    'RIGID.CONVNEXT.05': lambda: RIGIDConvNeXtHistogram(noise_level=0.05),
    'RIGID.CONVNEXT.10': lambda: RIGIDConvNeXtHistogram(noise_level=0.10),
    'RIGID.CONVNEXT.20': lambda: RIGIDConvNeXtHistogram(noise_level=0.20),
    'RIGID.CONVNEXT.30': lambda: RIGIDConvNeXtHistogram(noise_level=0.30),
    'RIGID.CONVNEXT.50': lambda: RIGIDConvNeXtHistogram(noise_level=0.50),
    'RIGID.CONVNEXT.75': lambda: RIGIDConvNeXtHistogram(noise_level=0.75),
    'RIGID.CONVNEXT.100': lambda: RIGIDConvNeXtHistogram(noise_level=1.0),

    # ===== ResNet-50 =====
    'RIGID.RESNET.001': lambda: RIGIDResNet50Histogram(noise_level=0.001),
    'RIGID.RESNET.01': lambda: RIGIDResNet50Histogram(noise_level=0.01),
    'RIGID.RESNET.05': lambda: RIGIDResNet50Histogram(noise_level=0.05),
    'RIGID.RESNET.10': lambda: RIGIDResNet50Histogram(noise_level=0.10),
    'RIGID.RESNET.20': lambda: RIGIDResNet50Histogram(noise_level=0.20),
    'RIGID.RESNET.30': lambda: RIGIDResNet50Histogram(noise_level=0.30),
    'RIGID.RESNET.50': lambda: RIGIDResNet50Histogram(noise_level=0.50),
    'RIGID.RESNET.75': lambda: RIGIDResNet50Histogram(noise_level=0.75),
    'RIGID.RESNET.100': lambda: RIGIDResNet50Histogram(noise_level=1.0),
}


def get_histogram_generator(statistic: str) -> Optional[object]:
    """Returns the appropriate histogram generator based on statistic name and level."""
    return STATISTIC_HISTOGRAMS.get(statistic)()
