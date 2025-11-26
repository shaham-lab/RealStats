"""FGSM attack experiment script (single fake sample) at the repository root."""

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torchvision import transforms
from torchvision.utils import save_image

mp.set_start_method("spawn", force=True)

from datasets_factory import DatasetFactory, DatasetType
from data_utils import ImageDataset
from processing.rigid_histogram import RIGIDCLSHistogram, RIGIDNormHistogram
from statistics_factory import get_histogram_generator
from stat_test import (
    TestType,
    calculate_pvals_from_cdf,
    compute_mean_std_dict,
    get_unique_id,
    interpret_keys_to_combinations,
    patch_parallel_preprocess,
)
from utils import compute_cdf, remove_nans_from_tests, set_seed


@dataclass
class AttackResult:
    """Container for attack evaluation results."""

    baseline_pvalue: float
    attacked_pvalue: float
    baseline_statistic: float
    attacked_statistic: float
    reference_mean: float
# --- Utility functions -----------------------------------------------------


def build_independent_keys(statistic: str, patch_size: int, seed: int) -> List[str]:
    return [get_unique_id(patch_size, statistic, seed)]


def resolve_stat_key(container: dict, independent_keys: List[str]) -> str:
    for key in independent_keys:
        if key in container:
            return key
        candidate = f"{key}_mean"
        if candidate in container:
            return candidate
    return next(iter(container.keys()))


def compute_reference_distributions(
    reference_dataset,
    independent_keys: List[str],
    batch_size: int,
    max_workers: int,
    num_data_workers: int,
    pkl_dir: str,
    seed: int,
    cache_suffix: str = "",
):
    combinations = interpret_keys_to_combinations(independent_keys)
    histograms = patch_parallel_preprocess(
        reference_dataset,
        batch_size,
        combinations,
        max_workers,
        num_data_workers,
        pkl_dir=pkl_dir,
        seed=seed,
        cache_suffix=cache_suffix,
    )
    histograms = compute_mean_std_dict(histograms)
    histograms = remove_nans_from_tests(histograms)
    histograms = {k: v for k, v in histograms.items() if k in independent_keys or k.rstrip("_mean") in independent_keys}

    cdfs = {test_id: compute_cdf(values, bins=500, test_id=test_id) for test_id, values in histograms.items()}
    return histograms, cdfs


def evaluate_single_image_pvalue(
    primary_image: torch.Tensor,
    primary_label: int,
    independent_keys: List[str],
    reference_cdfs,
    batch_size: int,
    max_workers: int,
    num_data_workers: int,
    pkl_dir: str,
    seed: int,
    transform: transforms.Compose,
    cache_suffix: str = "",
    cache_dir: Optional[str] = None,
) -> tuple[float, float]:
    combinations = interpret_keys_to_combinations(independent_keys)
    cache_root = cache_dir or os.path.join(pkl_dir, "fgsm_tmp")
    os.makedirs(cache_root, exist_ok=True)

    primary_path = os.path.join(cache_root, f"sample_primary_{cache_suffix or 'orig'}.png")

    save_image(primary_image, primary_path)

    dataset = ImageDataset(
        [primary_path],
        [primary_label],
        transform=transform,
    )
    histogram = patch_parallel_preprocess(
        dataset,
        batch_size,
        combinations,
        max_workers,
        num_data_workers,
        pkl_dir=pkl_dir,
        seed=seed,
        cache_suffix=cache_suffix,
    )
    histogram = compute_mean_std_dict(histogram)
    histogram = {
        k: np.ravel(np.asarray(v)[0] if np.asarray(v).ndim > 0 else np.asarray(v))
        for k, v in histogram.items()
        if k in independent_keys or k.rstrip("_mean") in independent_keys
    }
    pvalues = calculate_pvals_from_cdf(reference_cdfs, histogram, TestType.BOTH)
    pvalues = np.clip(np.array(pvalues, dtype=np.float32), 0.0, 1.0)

    # Recover the statistic value for logging purposes
    stat_key = next(iter(histogram.keys()))
    stat_value = float(np.squeeze(histogram[stat_key]))

    return float(pvalues.squeeze()), stat_value


# --- FGSM helpers ---------------------------------------------------------


def _forward_statistic(
    histogram_generator: RIGIDCLSHistogram,
    image_batch: torch.Tensor,
):
    """Run the RIGID statistic forward pass with gradients enabled."""

    # The processor expects a 4D BCHW tensor; squeeze or unsqueeze to normalize shapes.
    if image_batch.dim() > 4:
        image_batch = image_batch.squeeze(0)
    if image_batch.dim() == 3:
        image_batch = image_batch.unsqueeze(0)

    processor_inputs = histogram_generator.processor(images=image_batch, return_tensors="pt").to(
        histogram_generator.device
    )
    pixel_values = processor_inputs["pixel_values"].requires_grad_(True)

    outputs = histogram_generator.model(**{"pixel_values": pixel_values})
    embedding = histogram_generator.get_embedding(outputs)
    if embedding.dim() == 3:
        embedding = embedding[:, 0, :]

    noisy_pixels = torch.clamp(pixel_values + torch.randn_like(pixel_values) * histogram_generator.noise_level, 0, 1)
    noisy_outputs = histogram_generator.model(**{"pixel_values": noisy_pixels})
    noisy_embedding = histogram_generator.get_embedding(noisy_outputs)
    if noisy_embedding.dim() == 3:
        noisy_embedding = noisy_embedding[:, 0, :]

    similarity = F.cosine_similarity(embedding, noisy_embedding, dim=-1)
    return similarity.mean(), pixel_values


def iterative_fgsm_attack_statistic(
    histogram_generator: RIGIDCLSHistogram,
    image: torch.Tensor,
    epsilon: float,
    target_value: float,
    iterations: int,
) -> torch.Tensor:
    """Iteratively apply FGSM steps to drive the statistic toward the target value."""

    adv_image = image.detach().clone().to(histogram_generator.device)

    for _ in range(max(1, iterations)):
        image_batch = adv_image.unsqueeze(0)
        histogram_generator.model.zero_grad(set_to_none=True)
        stat_value, pixel_values = _forward_statistic(histogram_generator, image_batch)
        loss = (stat_value - target_value) ** 2
        loss.backward()

        with torch.no_grad():
            perturbation = epsilon * pixel_values.grad.sign()
            adv_image = torch.clamp(pixel_values - perturbation, 0, 1).detach()

    return adv_image.cpu().squeeze(0)


# --- Main experiment driver ----------------------------------------------


def run_attack_experiment(args) -> AttackResult:
    set_seed(args.seed)

    transform = transforms.Compose([transforms.Resize((args.sample_size, args.sample_size)), transforms.ToTensor()])
    datasets = DatasetFactory.create_dataset(dataset_type=args.dataset_type, transform=transform)
    reference_dataset = datasets["reference_real"]
    fake_dataset = datasets["test_fake"]

    independent_keys = build_independent_keys(args.statistic, args.patch_size, args.seed)

    reference_histograms, reference_cdfs = compute_reference_distributions(
        reference_dataset,
        independent_keys,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        num_data_workers=args.num_data_workers,
        pkl_dir=args.pkl_dir,
        seed=args.seed,
        cache_suffix=args.cache_suffix,
    )

    reference_key = resolve_stat_key(reference_histograms, independent_keys)
    reference_mean = float(np.mean(reference_histograms[reference_key]))

    fake_image, label, image_path = fake_dataset[args.fake_index]
    while fake_image.dim() > 3:
        fake_image = fake_image.squeeze(0)
    os.makedirs(args.output_dir, exist_ok=True)
    save_image(fake_image, os.path.join(args.output_dir, "original_fake.png"))

    baseline_pvalue, baseline_statistic = evaluate_single_image_pvalue(
        fake_image,
        label,
        independent_keys,
        reference_cdfs,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        num_data_workers=args.num_data_workers,
        pkl_dir=args.pkl_dir,
        seed=args.seed,
        transform=transform,
        cache_suffix=f"baseline_{args.cache_suffix}",
        cache_dir=args.output_dir,
    )

    histogram_generator = get_histogram_generator(args.statistic)
    if not isinstance(histogram_generator, (RIGIDCLSHistogram, RIGIDNormHistogram)):
        raise ValueError("FGSM attack currently supports RIGID-based statistics only.")

    attacked_image = iterative_fgsm_attack_statistic(
        histogram_generator,
        fake_image,
        args.epsilon,
        reference_mean,
        args.iterations,
    )
    save_image(attacked_image, os.path.join(args.output_dir, "attacked_fake.png"))

    attacked_pvalue, attacked_statistic = evaluate_single_image_pvalue(
        attacked_image,
        label,
        independent_keys,
        reference_cdfs,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        num_data_workers=args.num_data_workers,
        pkl_dir=args.pkl_dir,
        seed=args.seed,
        transform=transform,
        cache_suffix=f"attacked_{args.cache_suffix}",
        cache_dir=args.output_dir,
    )

    result = AttackResult(
        baseline_pvalue=baseline_pvalue,
        attacked_pvalue=attacked_pvalue,
        baseline_statistic=baseline_statistic,
        attacked_statistic=attacked_statistic,
        reference_mean=reference_mean,
    )

    with open(os.path.join(args.output_dir, "fgsm_attack_results.json"), "w") as f:
        json.dump(result.__dict__, f, indent=2)

    print("[INFO] FGSM attack completed")
    print(f"  Statistic: {args.statistic}")
    print(f"  Attack iterations: {args.iterations}")
    print(f"  Target reference mean: {reference_mean:.6f}")
    print(f"  Baseline statistic value: {baseline_statistic:.6f}")
    print(f"  Attacked statistic value: {attacked_statistic:.6f}")
    print(f"  Baseline p-value: {baseline_pvalue:.6f}")
    print(f"  Attacked p-value: {attacked_pvalue:.6f}")
    print(f"  Fake sample path: {image_path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FGSM attack against a single statistic.")
    parser.add_argument("--statistic", type=str, default="RIGID.DINO.05", help="Statistic to attack (e.g., RIGID.DINO.05)")
    parser.add_argument("--epsilon", type=float, default=1e-3, help="FGSM step size")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterative FGSM steps")
    parser.add_argument("--dataset_type", type=str, default="ALL", choices=[e.name for e in DatasetType], help="Dataset split")
    parser.add_argument("--fake_index", type=int, default=0, help="Index of fake sample to attack")
    parser.add_argument("--sample_size", type=int, default=512, help="Image resize before processing")
    parser.add_argument("--patch_size", type=int, default=512, help="Patch size for statistic computation")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for preprocessing")
    parser.add_argument("--num_data_workers", type=int, default=2, help="Dataloader workers")
    parser.add_argument("--max_workers", type=int, default=2, help="Multiprocessing workers")
    parser.add_argument("--pkl_dir", type=str, default="pkls/AIStats/new_stats", help="Cache directory for statistics")
    parser.add_argument("--output_dir", type=str, default="outputs/fgsm_attack", help="Directory to save artifacts")
    parser.add_argument("--cache_suffix", type=str, default="fgsm", help="Suffix for cache isolation")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    attack_args = parser.parse_args()
    run_attack_experiment(attack_args)
