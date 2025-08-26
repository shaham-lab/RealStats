import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from processing.histograms import BaseHistogram
from transformers import AutoImageProcessor, CLIPModel
from transformers import AutoImageProcessor, ResNetForImageClassification

class RIGIDNormHistogram(BaseHistogram):
    def __init__(self, model_name, noise_level=0.05):
        super().__init__()
        self.model_name = model_name
        self.noise_level = noise_level

        self.processor = self.load_processor()
        self.model = self.load_model().to(self.device).eval()

    def load_processor(self):
        """Must be implemented by subclass."""
        raise NotImplementedError

    def load_model(self):
        """Must be implemented by subclass."""
        raise NotImplementedError

    def extract_features(self, image_batch):
        """Extract features from the image batch."""
        with torch.no_grad():
            inputs = self.processor(images=image_batch, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            return self.get_embedding(outputs)

    def get_embedding(self, outputs):
        """Must be implemented by subclass. Extracts embedding tensor from model outputs."""
        raise NotImplementedError

    def add_noise(self, image_batch):
        noise = torch.randn_like(image_batch) * self.noise_level
        return torch.clamp(image_batch + noise, 0, 1)

    def preprocess(self, image_batch):
        with torch.no_grad():
            original = self.extract_features(image_batch)
            perturbed = self.extract_features(self.add_noise(image_batch))

        similarity = torch.norm(F.cosine_similarity(original, perturbed, dim=-1), p=2, dim=1).cpu().numpy()
        return similarity


class RIGIDCLSHistogram(BaseHistogram):
    def __init__(self, model_name, noise_level=0.05):
        super().__init__()
        self.model_name = model_name
        self.noise_level = noise_level

        self.processor = self.load_processor()
        self.model = self.load_model().to(self.device).eval()

    def load_processor(self):
        """Must be implemented by subclass."""
        raise NotImplementedError

    def load_model(self):
        """Must be implemented by subclass."""
        raise NotImplementedError

    def extract_features(self, image_batch):
        """Extract features from the image batch."""
        with torch.no_grad():
            inputs = self.processor(images=image_batch, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            return self.get_embedding(outputs)[:, 0, :]

    def get_embedding(self, outputs):
        """Must be implemented by subclass. Extracts embedding tensor from model outputs."""
        raise NotImplementedError

    def add_noise(self, image_batch):
        noise = torch.randn_like(image_batch) * self.noise_level
        return torch.clamp(image_batch + noise, 0, 1)

    def preprocess(self, image_batch):
        with torch.no_grad():
            original = self.extract_features(image_batch)
            perturbed = self.extract_features(self.add_noise(image_batch))

        similarity = F.cosine_similarity(original, perturbed, dim=-1).cpu().numpy()
        return similarity


class RIGIDCLSMultiplePermutationsHistogram(BaseHistogram):
    def __init__(self, model_name, noise_level=0.05, num_noises=5, weights=None):
        """
        Args:
            model_name (str): Model identifier.
            noise_level (float): Standard deviation of noise to add.
            num_noises (int): Number of different noise perturbations.
            weights (list or None): Optional weights for each noise permutation.
        """
        super().__init__()
        self.model_name = model_name
        self.noise_level = noise_level
        self.num_noises = num_noises

        self.weights = (
            torch.tensor(weights, dtype=torch.float32)
            if weights is not None
            else torch.ones(num_noises) / num_noises
        ).to(self.device)

        self.processor = self.load_processor()
        self.model = self.load_model().to(self.device).eval()

    def load_processor(self):
        """Must be implemented by subclass."""
        raise NotImplementedError

    def load_model(self):
        """Must be implemented by subclass."""
        raise NotImplementedError

    def get_embedding(self, outputs):
        """Must be implemented by subclass. Extracts embedding tensor from model outputs."""
        raise NotImplementedError

    def extract_features(self, image_batch):
        """Extract features from the image batch."""
        with torch.no_grad():
            inputs = self.processor(images=image_batch, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            return self.get_embedding(outputs)[:, 0, :]  # (B, D)

    def add_noise_variants(self, image_batch):
        """
        Efficiently generate multiple noisy variants of image_batch.

        Args:
            image_batch (torch.Tensor): (B, C, H, W)
        Returns:
            torch.Tensor: Noisy variants of shape (N * B, C, H, W)
        """
        B, C, H, W = image_batch.shape

        # Repeat images N times
        repeated = image_batch.unsqueeze(0).repeat(self.num_noises, 1, 1, 1, 1)

        # Generate and add noise
        noise = torch.randn_like(repeated) * self.noise_level
        noisy_images = torch.clamp(repeated + noise, 0, 1)

        return noisy_images.view(-1, C, H, W)  # Shape: (N * B, C, H, W)

    def preprocess(self, image_batch):
        """
        Extract features from original and linearly combined noisy versions,
        and return cosine similarity.
        """
        with torch.no_grad():
            B = image_batch.size(0)

            # Original embeddings: (B, D)
            original = self.extract_features(image_batch)

            # Generate all noisy variants at once: (N * B, C, H, W)
            noisy_batch = self.add_noise_variants(image_batch)

            # Extract embeddings for all noisy variants: (N * B, D)
            all_embeddings = self.extract_features(noisy_batch)

            # Reshape to (N, B, D)
            noisy_embeddings = all_embeddings.view(self.num_noises, B, -1)

            # Apply weights: (N, 1, 1) * (N, B, D) → (N, B, D)
            weighted = noisy_embeddings * self.weights.view(-1, 1, 1)

            # Weighted sum over noise dimension: (B, D)
            combined = torch.sum(weighted, dim=0)

            # Cosine similarity between original and combined: (B,)
            similarity = F.cosine_similarity(original, combined, dim=-1)

            return similarity.cpu().numpy()


class RIGIDDinoV2Histogram(RIGIDCLSHistogram):
    def __init__(self, noise_level=0.05):
        super().__init__(model_name="facebook/dinov2-large", noise_level=noise_level)

    def load_processor(self):
        return AutoImageProcessor.from_pretrained(self.model_name, do_rescale=False, use_fast=True)

    def load_model(self):
        return AutoModel.from_pretrained(self.model_name)

    def get_embedding(self, outputs):
        return outputs.last_hidden_state  # shape: [B, seq_len, dim]


class RIGIDDinoV3Histogram(RIGIDCLSHistogram):
    """RIGID statistic using DINOv3 vision backbone."""

    def __init__(self, model_name="facebook/dinov3-vits16-pretrain-lvd1689m", noise_level=0.05):
        super().__init__(model_name=model_name, noise_level=noise_level)

    def load_processor(self):
        return AutoImageProcessor.from_pretrained(self.model_name, do_rescale=False, use_fast=True)

    def load_model(self):
        return AutoModel.from_pretrained(self.model_name)

    def get_embedding(self, outputs):
        return outputs.last_hidden_state  # shape: [B, seq_len, dim]
    
class RIGIDEVAHistogram(RIGIDCLSHistogram):
    def __init__(self, noise_level=0.05):
        super().__init__(model_name="eva_clip_g/eva02-large-336", noise_level=noise_level)

    def load_processor(self):
        return AutoImageProcessor.from_pretrained(self.model_name, do_rescale=False, use_fast=True)

    def load_model(self):
        return AutoModel.from_pretrained(self.model_name)

    def get_embedding(self, outputs):
        return outputs.last_hidden_state  # [B, tokens, dim]

class RIGIDBEiTHistogram(RIGIDCLSHistogram):
    def __init__(self, noise_level=0.05):
        super().__init__(model_name="microsoft/beit-large-patch16-224", noise_level=noise_level)

    def load_processor(self):
        return AutoImageProcessor.from_pretrained(self.model_name, do_rescale=False, use_fast=True)

    def load_model(self):
        return AutoModel.from_pretrained(self.model_name)

    def get_embedding(self, outputs):
        return outputs.last_hidden_state


class RIGIDOpenCLIPHistogram(RIGIDCLSHistogram):
    def __init__(self, noise_level=0.05):
        super().__init__(model_name="laion/CLIP-ViT-H-14-laion2B-s32B-b79K", noise_level=noise_level)

    def load_processor(self):
        return AutoImageProcessor.from_pretrained(self.model_name, do_rescale=False, use_fast=True)

    def load_model(self):
        model = CLIPModel.from_pretrained(self.model_name)
        return model.vision_model  # ✅ Use only the vision tower

    def get_embedding(self, outputs):
        return outputs.last_hidden_state  # [B, seq_len, dim]


class RIGIDOpenAICLIPHistogram(RIGIDCLSHistogram):
    def __init__(self, noise_level=0.05):
        super().__init__(model_name="openai/clip-vit-large-patch14", noise_level=noise_level)

    def load_processor(self):
        return AutoImageProcessor.from_pretrained(self.model_name, do_rescale=False, use_fast=True)

    def load_model(self):
        model = CLIPModel.from_pretrained(self.model_name)
        return model.vision_model  # Use only the vision tower

    def get_embedding(self, outputs):
        return outputs.last_hidden_state  # [B, seq_len, dim]


class RIGIDBigGCLIPHistogram(RIGIDCLSHistogram):
    def __init__(self, noise_level=0.05):
        super().__init__(model_name="laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", noise_level=noise_level)

    def load_processor(self):
        return AutoImageProcessor.from_pretrained(self.model_name, do_rescale=False, use_fast=True)

    def load_model(self):
        model = CLIPModel.from_pretrained(self.model_name)
        return model.vision_model

    def get_embedding(self, outputs):
        return outputs.last_hidden_state


class RIGIDConvNeXtHistogram(RIGIDCLSHistogram):
    def __init__(self, noise_level=0.05):
        super().__init__(model_name="facebook/convnext-base-224", noise_level=noise_level)

    def load_processor(self):
        return AutoImageProcessor.from_pretrained(self.model_name, do_rescale=False, use_fast=True)

    def load_model(self):
        return AutoModel.from_pretrained(self.model_name)

    def get_embedding(self, outputs):
        # outputs.last_hidden_state: [B, C, H, W]
        x = outputs.last_hidden_state
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))  # → [B, C, 1, 1]
        x = x.view(x.size(0), 1, -1)  # → [B, 1, C]
        return x


class RIGIDResNet50Histogram(RIGIDCLSHistogram):
    def __init__(self, noise_level=0.05):
        super().__init__(model_name="microsoft/resnet-50", noise_level=noise_level)

    def load_processor(self):
        return AutoImageProcessor.from_pretrained(self.model_name, do_rescale=False, use_fast=True)

    def load_model(self):
        return AutoModel.from_pretrained(self.model_name)  # Loads ResNetModel (no classifier head)

    def get_embedding(self, outputs):
        # outputs.last_hidden_state shape: [B, C, H, W]
        x = outputs.last_hidden_state
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))  # [B, C, 1, 1]
        x = x.view(x.size(0), 1, -1)  # [B, 1, C]
        return x
