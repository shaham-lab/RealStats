from diffusers import DDPMScheduler, StableDiffusionPipeline
import os
import torch
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms.functional as F
from transformers import AutoImageProcessor, CLIPModel, pipeline as pipeline_caption
import pandas as pd
from pathlib import Path
from typing import Dict, Iterable, List, Union
from tqdm import tqdm

from processing.histograms import BaseHistogram


def clip_preprocess(img_t, siz, do_resize=False):
    """
    Converts VAE-decoded images to [0,1] and ensures compatibility with CLIP input.
    """
    if img_t.shape[-1] and do_resize != siz:
        img_t = F.resize(img_t, [siz, siz])  # Resize to target size if needed

    # ✅ Convert from [-float,float] → [0,1]
    img_t = (img_t / 2 + 0.5).clamp(0, 1).float()

    return img_t


def get_sd_v14_model(device):
    """
    Loads the Stable Diffusion v1.4 model and its components.

    Args:
        device (str): The device to load the model onto ('cuda' or 'cpu').

    Returns:
        tuple: (unet, tokenizer, text_encoder, vae, scheduler)
    """
    pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"
    pipeline = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch.float16)
    pipeline.to(device)

    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    unet = pipeline.unet.to(device).eval().half()
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder.to(device).half()
    vae = pipeline.vae.to(device).eval().half()

    return unet, tokenizer, text_encoder, vae, noise_scheduler


def load_sdv14_criterion_functionalities(device):
    """
    Loads all necessary functionalities for SD v1.4-based criteria.

    Args:
        device (str): The device to load the models onto ('cuda' or 'cpu').

    Returns:
        tuple: (unet, tokenizer, text_encoder, vae, scheduler, cos, clip, processor, image_to_text)
    """
    # Load cosine similarity function
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6).to(device)

    # Load SD v1.4 model components
    unet, tokenizer, text_encoder, vae, scheduler = get_sd_v14_model(device)

    # Load image-to-text model (LLaVA)
    image_to_text = pipeline_caption("image-to-text", model="llava-hf/llava-1.5-7b-hf", device=device)

    # Load CLIP model and processor
    processor = AutoImageProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=True)
    clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)

    return unet, tokenizer, text_encoder, vae, scheduler, cos, clip, processor, image_to_text


class LatentNoiseCriterion(BaseHistogram):
    """Computes noise-based criteria in latent space using Stable Diffusion and CLIP."""

    def __init__(self, device="cuda", num_noise=8, epsilon_reg=1e-6, siz=512, time_frac=0.01):
        """
        Initialize the latent noise criterion class.

        :param vae: Stable Diffusion VAE model.
        :param unet: UNet model for noise prediction.
        :param text_encoder: CLIP text encoder.
        :param tokenizer: Tokenizer for text processing.
        :param scheduler: Diffusion noise scheduler.
        :param clip: CLIP model for feature extraction.
        :param processor: CLIP processor for image embeddings.
        :param device: Device (CPU or GPU).
        :param num_noise: Number of noise samples per image.
        :param epsilon_reg: Small constant to avoid division by zero.
        """
        super().__init__()
        unet, tokenizer, text_encoder, vae, scheduler, cos, clip_model, clip_processor, image_to_text = load_sdv14_criterion_functionalities(device)

        self.vae = vae.eval().to(device).half()
        self.unet = unet.eval().to(device).half()
        self.text_encoder = text_encoder.eval().to(device).half()
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.clip = clip_model.eval().to(device).half()
        self.processor = clip_processor
        self.device = device
        self.num_noise = num_noise
        self.epsilon_reg = epsilon_reg
        self.siz = siz
        self.time_frac = time_frac

    def preprocess(self, images_raw, prompts_list=None, return_terms=False):
        """
        Main function to compute the criterion based on image noise characteristics.

        :param images_raw: List of input images (batch format).
        :param siz: Target image size for preprocessing.
        :param prompts_list: Optional text prompts for conditioning.
        :param time_frac: Noise fraction for diffusion step.
        :param return_terms: If True, returns additional terms (bias, kappa, D).
        :return: List of dictionaries with computed statistics.
        """
        num_images = len(images_raw)
        outputs = []

        for i in range(num_images):
            # Step 1: Preprocess Images
            image_raw = images_raw[i].unsqueeze(0)  # Add batch dimension

            images = self.preprocess_images(image_raw)

            # Step 2: Encode Text Prompts
            text_emb = self.encode_text(prompts_list, len(image_raw))

            # Step 3: Encode Images into Latent Space
            latents = self.encode_latents(images)

            # Step 4: Generate and Apply Spherical Noise
            spherical_noise = self.generate_spherical_noise(latents)
            noisy_latents, timestep = self.add_noise_to_latents(latents, spherical_noise, self.time_frac)

            # Step 5: Predict Noise using UNet
            noise_pred = self.predict_noise(noisy_latents, timestep, text_emb)

            # Step 6: Decode Images from Latents
            decoded_noise = self.decode_latents(noise_pred)
            decoded_spherical_noise = self.decode_latents(spherical_noise)

            # Step 6.5: Fix values: 
            decoded_noise_fixed = clip_preprocess(decoded_noise, siz=self.siz, do_resize=True)
            decoded_spherical_noise_fixed = clip_preprocess(decoded_spherical_noise, siz=self.siz, do_resize=True)

            # Step 7: Compute CLIP Similarities
            img_clip = self.compute_clip_embeddings(image_raw)
            img_d_clip = self.compute_clip_embeddings(decoded_noise_fixed)
            img_s_clip = self.compute_clip_embeddings(decoded_spherical_noise_fixed)

            # Step 8: Compute Statistics & Criterion
            stats = self.compute_statistics(img_clip, img_d_clip, img_s_clip)
            criterion = self.compute_criterion(stats)

            # Step 9: Return Results
            result = {"criterion": float(criterion)}
            if return_terms:
                result.update(stats)
            outputs.append(criterion)
        return np.array(outputs)

    ### **Helper Functions for Each Step**
    def preprocess_images(self, image_batch):
        """ Resize and normalize images for model input. """
        # return preprocess_image_batch(image_batch, self.siz, self.device)
        return image_batch.half() * 2 - 1

    def encode_text(self, prompts_list, num_images):
        """ Encode text prompts into embeddings. """
        if prompts_list is not None:
            prompts = prompts_list * num_images if len(prompts_list) == 1 else prompts_list
        else:
            prompts = [""] * num_images  # Default to empty prompts

        expanded_prompts = [p for p in prompts for _ in range(self.num_noise)]
        text_tokens = self.tokenizer(expanded_prompts, padding="max_length", max_length=77, truncation=True, return_tensors="pt")
        input_ids = text_tokens.input_ids.to(self.device)
        return self.text_encoder(input_ids).last_hidden_state

    def encode_latents(self, images):
        """ Encode images into latent space using VAE. """
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents *= self.vae.config.scaling_factor
        return latents.repeat_interleave(self.num_noise, dim=0).half()

    def generate_spherical_noise(self, latents):
        """ Generate normalized spherical noise. """
        gauss_noise = torch.randn_like(latents, device=self.device).half()
        norm = torch.norm(gauss_noise, p=2, dim=(1, 2, 3), keepdim=True)
        spherical_noise = gauss_noise / (norm + self.epsilon_reg)

        sqrt_d = torch.prod(torch.tensor(latents.shape[1:])).float().sqrt()
        return spherical_noise * sqrt_d

    def add_noise_to_latents(self, latents, spherical_noise, time_frac):
        """ Add noise to latents using the diffusion scheduler. """
        timestep = time_frac * self.scheduler.config.num_train_timesteps
        timestep_tensor = torch.full((latents.shape[0],), timestep, device=self.device, dtype=torch.long)
        return self.scheduler.add_noise(
            original_samples=latents,
            noise=spherical_noise,
            timesteps=timestep_tensor,
        ).half(), timestep_tensor


    def predict_noise(self, noisy_latents, timestep, text_emb):
        """ Predict noise using UNet model. """
        with torch.no_grad():
            noise_pred = self.unet(noisy_latents, timestep, encoder_hidden_states=text_emb)[0]
        return 1 / self.vae.config.scaling_factor * noise_pred

    def decode_latents(self, latents, sub_batch_size=16):
        """ Decode latent vectors into images. """
        num_samples = latents.shape[0]

        if num_samples <= sub_batch_size and False:
            return self.vae.decode(latents, return_dict=False)[0]
        with torch.no_grad():
            decoded_list = []
            for i in range(0, num_samples, sub_batch_size):
                batch = latents[i:i+sub_batch_size]
                torch.cuda.empty_cache()
                decoded_list.append(self.vae.decode(batch, return_dict=False)[0])

        return torch.cat(decoded_list, dim=0)

    def compute_clip_embeddings(self, images):
        """ Compute CLIP feature embeddings for images. """
        img_in = self.processor(images=images, return_tensors="pt").to(self.device)
        return self.clip.get_image_features(**img_in).detach().cpu()

    def compute_statistics(self, img_clip, img_d_clip, img_s_clip):
        """ Compute similarity statistics using cosine similarity. """
        cos = torch.nn.CosineSimilarity(dim=1, eps=self.epsilon_reg).to(self.device)

        if img_clip.shape[0] != img_d_clip.shape[0]:
            img_clip = img_clip.repeat_interleave(img_d_clip.shape[0] // img_clip.shape[0], dim=0)

        bias_vec = cos(img_clip, img_d_clip).numpy()
        kappa_vec = cos(img_d_clip, img_s_clip).numpy()
        D_vec = torch.norm(img_d_clip.view(img_d_clip.size(0), -1), p=2, dim=1).cpu().numpy()

        return {
            "bias_mean": bias_vec.mean(),
            "kappa_mean": kappa_vec.mean(),
            "D_mean": D_vec.mean()
        }

    def compute_criterion(self, stats):
        """ Compute the final criterion score. """
        d_clip = 512
        sqrt_d_clip = d_clip ** 0.5
        return 1 + (sqrt_d_clip * stats["bias_mean"] - stats["D_mean"] + stats["kappa_mean"]) / (sqrt_d_clip + 2)


class PathBasedStatistic(BaseHistogram):
    """Utility base class for statistics that rely on sample metadata instead of pixels."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # These statistics do not need a GPU device.
        self.device = torch.device("cpu")

    def preprocess(self, image_batch):
        raise NotImplementedError("PathBasedStatistic does not process image batches.")

    def lookup(self, paths: Iterable[str]) -> List[float]:
        """Return the statistic for each input path."""
        raise NotImplementedError

    def create_histogram(self, data_loader):
        results: Dict[str, float] = {}

        for _, _, paths in tqdm(data_loader, desc="Generating histograms", leave=False):
            scores = self.lookup(paths)
            for path, score in zip(paths, scores):
                results[path] = float(score)

        return results


class LatentNoiseCriterionOriginal(PathBasedStatistic):
    """Loads pre-computed Latent Noise Criterion scores from disk."""

    def __init__(self, scores_csv: Union[str, Path, None] = None):
        super().__init__()
        if scores_csv is None:
            scores_csv = os.getenv("LATENT_NOISE_CRITERION_ORIGINAL_CSV")
            if scores_csv is None:
                raise ValueError(
                    "Path to the LatentNoiseCriterion_original CSV must be provided via "
                    "the constructor or the LATENT_NOISE_CRITERION_ORIGINAL_CSV environment variable."
                )

        self.scores_csv = Path(scores_csv).expanduser().resolve()
        if not self.scores_csv.exists():
            raise FileNotFoundError(f"Could not find CSV with scores at {self.scores_csv}")

        self._scores = self._load_scores(self.scores_csv)

    def _load_scores(self, csv_path: Path) -> Dict[str, float]:
        df = pd.read_csv(csv_path)
        if "image_path" not in df.columns or "criterion" not in df.columns:
            raise ValueError(
                "LatentNoiseCriterion_original CSV must contain 'image_path' and 'criterion' columns."
            )

        repo_root = Path(__file__).resolve().parents[1]
        mapping: Dict[str, float] = {}

        for _, row in df.iterrows():
            image_path = Path(str(row["image_path"])).expanduser()
            score = float(row["criterion"])

            # Store multiple keys for robust lookup (relative and absolute).
            candidates = {
                image_path,
                repo_root / image_path,
                (repo_root / image_path).resolve(),
            }

            for candidate in candidates:
                mapping[os.path.normpath(str(candidate))] = score

        return mapping

    def lookup(self, paths: Iterable[str]) -> List[float]:
        scores: List[float] = []
        repo_root = Path(__file__).resolve().parents[1]

        for path in paths:
            normalized = os.path.normpath(str(Path(path).expanduser().resolve()))
            if normalized in self._scores:
                scores.append(self._scores[normalized])
                continue

            # Fall back to using the repository-relative path if available.
            relative = normalized
            if Path(normalized).is_absolute():
                try:
                    relative = os.path.normpath(str(Path(normalized).relative_to(repo_root)))
                except ValueError:
                    relative = normalized
            score = self._scores.get(relative)
            if score is None:
                raise KeyError(
                    f"No pre-computed LatentNoiseCriterion score found for '{path}'."
                )
            scores.append(score)

        return scores
