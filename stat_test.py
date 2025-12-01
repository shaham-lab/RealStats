import itertools
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence
import random
import re
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from statistics_factory import get_histogram_generator
from processing.manifold_bias_histogram import PathBasedStatistic
from utils import (
    compute_cdf,
    compute_chi2_and_corr_matrix,
    compute_mean_std_dict,
    finding_optimal_independent_subgroup_deterministic,
    perform_ensemble_testing,
    remove_nans_from_tests,
    set_seed,
)
from data_utils import GlobalPatchDataset, SelfPatchDataset
from enum import Enum
import traceback
import gc


class TestType(Enum):
    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"


@dataclass
class ReferenceModel:
    independent_keys: List[str]
    reference_cdfs: Dict[str, np.ndarray]
    reference_cache_suffix: str = ""

    @property
    def independent_combinations(self):
        return interpret_keys_to_combinations(self.independent_keys)


def get_unique_id(patch_size, statistic, seed=42):
    """Generate a unique ID for processing."""
    return f"PatchProcessing_statistic={statistic}_patch_size={patch_size}_seed={seed}"


def preprocess_statistic(dataset, batch_size, statistic, num_data_workers, patch_size, pkl_dir, seed=42, cache_suffix=""):
    """Preprocess the dataset for a single statistic name using various histogram statistics."""
    set_seed(seed)

    unique_id = get_unique_id(patch_size, statistic, seed)
    combo_dir_name = f"{unique_id}{cache_suffix}" if cache_suffix else unique_id
    combo_dir = os.path.join(pkl_dir, combo_dir_name)
    results = []

    expected_stat_paths = [
        os.path.join(combo_dir, os.path.splitext(p.lstrip(os.sep))[0] + ".npy")
        for p in dataset.image_paths
    ]

    if all(os.path.exists(p) for p in expected_stat_paths):
        first = np.load(expected_stat_paths[0], mmap_mode=None)
        results = np.empty((len(expected_stat_paths),) + first.shape, dtype=first.dtype)
        results[0] = first
        for i, p in enumerate(expected_stat_paths[1:], 1):
            results[i] = np.load(p, mmap_mode=None)
        assert results.shape[0] == len(expected_stat_paths), f"Expected {len(expected_stat_paths)} samples, got {results.shape[0]}"
        return results

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_data_workers)

    histogram_generator = get_histogram_generator(statistic)
    if histogram_generator is None:
        return None

    is_path_based = isinstance(histogram_generator, PathBasedStatistic)

    for images, _, paths in tqdm(data_loader, desc=f"Processing {statistic}", unit="batch", total=len(data_loader), leave=False):
        B, P = images.shape[:2]

        cached = [None] * B
        to_compute = []

        for i, path in enumerate(paths):
            stat_path = os.path.join(combo_dir, os.path.splitext(path.lstrip(os.sep))[0] + ".npy")
            if os.path.exists(stat_path):
                cached[i] = np.load(stat_path, mmap_mode=None)
            else:
                to_compute.append((i, path, stat_path))

        if to_compute:
            idxs = [i for i, _, _ in to_compute]

            if is_path_based:
                lookup_paths = [path for _, path, _ in to_compute]
                scores = histogram_generator.lookup(lookup_paths)
                hist = np.repeat(np.asarray(scores, dtype=np.float32)[:, None], P, axis=1)
            else:
                imgs = images[idxs].view(len(idxs) * P, *images.shape[2:])
                imgs = imgs.to(histogram_generator.device)
                hist = histogram_generator.preprocess(imgs)
                hist = hist.reshape(len(idxs), P)

            for (i, _, stat_path), h in zip(to_compute, hist):
                os.makedirs(os.path.dirname(stat_path), exist_ok=True)
                np.save(stat_path, h)
                cached[i] = h

        results.extend(cached)

    torch.cuda.empty_cache()
    gc.collect()
    stacked = np.stack(results, axis=0)

    assert stacked.shape[0] == len(expected_stat_paths), f"Expected {len(expected_stat_paths)} samples, got {stacked.shape[0]}"
    return stacked


def calculate_cdfs_pvalues(real_population_cdfs, input_samples_values, test_type=TestType.LEFT):
    """
    Calculate p-values for input samples against real population CDFs.
    """
    p_values_per_test = []
    for descriptor, sample in input_samples_values.items():
        _, bin_edges, population_cdf = real_population_cdfs[descriptor]
        
        bin_index = np.clip(np.digitize(sample, bin_edges) - 1, 0, len(population_cdf) - 1)
        pvalue = None 

        if test_type == TestType.LEFT:
            # Only compute left-tailed p-values
            pvalue = population_cdf[bin_index]

        elif test_type == TestType.RIGHT:
            # Only compute right-tailed p-values
            left_p_values = population_cdf[bin_index]
            pvalue = 1 - left_p_values

        elif test_type == TestType.BOTH:
            # Compute two-tailed p-values
            left_p_values = population_cdf[bin_index]
            right_p_values = 1 - left_p_values
            pvalue = np.minimum(left_p_values, right_p_values) * 2

        p_values_per_test.append(pvalue)

    return p_values_per_test


def validate_real_population_cdfs(real_population_cdfs, input_samples):
    """
    Ensure all descriptors in input samples are present in real_population_cdfs.
    """
    sample_descriptors = set(input_samples[0].keys())
    missing_descriptors = sample_descriptors - real_population_cdfs.keys()
    if missing_descriptors:
        raise KeyError(f"Descriptors missing in real_population_cdfs: {missing_descriptors}")


def calculate_pvals_from_cdf(real_population_cdfs, samples_histogram, test_type=TestType.LEFT):
    input_samples = [dict(zip(samples_histogram.keys(), values)) for values in zip(*samples_histogram.values())]
    validate_real_population_cdfs(real_population_cdfs, input_samples)
    input_samples_pvalues = [calculate_cdfs_pvalues(real_population_cdfs, sample, test_type) for sample in tqdm(input_samples, desc=f"Calculating P-values")]
    return input_samples_pvalues


def generate_combinations(patch_sizes, statistics):
    """Generate all combinations for the training phase."""
    combinations = []
    for patch_size, statistic in itertools.product(patch_sizes, statistics):
        combinations.append({
            'patch_size': patch_size,
            'statistic': statistic,
        })
    return combinations


def interpret_keys_to_combinations(independent_keys_group):
    """Convert independent keys to relevant test combinations and preserve order while removing duplicates."""
    seen = set()
    combinations = []

    for key in independent_keys_group:
        match = re.match(r"PatchProcessing_statistic=([\w.]+)_patch_size=(\d+)", key)
        if match:
            statistic, patch_size = match.groups()
            combination = {
                'statistic': statistic,
                'patch_size': int(patch_size),
            }
            key_tuple = (statistic, int(patch_size))
            if key_tuple not in seen:
                seen.add(key_tuple)
                combinations.append(combination)

    return combinations


def patch_parallel_preprocess(original_dataset, batch_size, combinations, max_workers, num_data_workers, pkl_dir='pkls', sort=True, seed=42, cache_suffix=""):
    """Preprocess the dataset for specific combinations in parallel."""
    # Stage 1.1: Multi-detector processing across detector configurations.
    results = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_combination = {
            executor.submit(preprocess_statistic, SelfPatchDataset(original_dataset, comb['patch_size']), batch_size, comb['statistic'], num_data_workers, comb['patch_size'], pkl_dir, seed, cache_suffix): comb
            for comb in combinations
        }

        for future in tqdm(as_completed(future_to_combination), total=len(future_to_combination), desc="Processing..."):
            combination = future_to_combination[future]
            unique_id = get_unique_id(combination['patch_size'], combination['statistic'], seed)
            try:
                results[unique_id] = future.result()
                
            except Exception as exc:
                error_msg = traceback.format_exc()
                print(f"Combination {combination}, generated an exception:\n{error_msg}")

        results = {k: v for k, v in results.items() if v is not None}

        if sort:
            results = {k: results[k] for k in sorted(results)}

    return results


def build_reference_model(
    reference_dataset,
    statistics: Sequence[str] | None,
    patch_sizes: Sequence[int] | None,
    batch_size: int,
    max_workers: int,
    num_data_workers: int,
    pkl_dir: str,
    seed: int,
    cdf_bins: int,
    ensemble_test: str,
    chi2_bins: int,
    ks_pvalue_abs_threshold: float,
    cremer_v_threshold: float,
    preferred_statistics: Iterable[str] | None = None,
    independent_keys: Sequence[str] | None = None,
    test_type: TestType = TestType.LEFT,
    logger=None,
    cache_suffix: str = "",
) -> ReferenceModel:
    """Calibrate detectors on the reference set and select independent statistics."""
    if independent_keys is not None:
        training_combinations = interpret_keys_to_combinations(independent_keys)
    else:
        if statistics is None or patch_sizes is None:
            raise ValueError("statistics and patch_sizes are required when independent_keys are not provided")
        training_combinations = generate_combinations(patch_sizes, statistics)

    # Stage 1.1: Multi-detector processing across detector configurations.
    reference_histogram = patch_parallel_preprocess(
        reference_dataset,
        batch_size,
        training_combinations,
        max_workers,
        num_data_workers,
        pkl_dir=pkl_dir,
        seed=seed,
        cache_suffix=cache_suffix,
    )
    reference_histogram = compute_mean_std_dict(reference_histogram)
    reference_histogram = remove_nans_from_tests(reference_histogram)

    # Stage 1.2: Empirical two-sided p-value modeling via ECDFs.
    reference_cdfs = {
        test_id: compute_cdf(values, bins=cdf_bins, test_id=test_id)
        for test_id, values in reference_histogram.items()
    }

    tuning_reference_pvals = calculate_pvals_from_cdf(reference_cdfs, reference_histogram, test_type)
    tuning_reference_pvals = np.clip(tuning_reference_pvals, 0, 1)

    keys = list(reference_histogram.keys())
    tuning_pvalue_distributions = tuning_reference_pvals.T

    if independent_keys is not None:
        independent_keys_group = list(independent_keys)
        best_results = {}
    else:
        # Stage 2.1: Pairwise dependence testing under the null.
        chi2_p_matrix, _ = compute_chi2_and_corr_matrix(
            keys, tuning_pvalue_distributions, max_workers=max_workers, bins=chi2_bins
        )

        # Stage 2.2-2.3: Independence graph construction and maximal clique selection.
        independent_keys_group, best_results, _ = finding_optimal_independent_subgroup_deterministic(
            keys=keys,
            chi2_p_matrix=chi2_p_matrix,
            pvals_matrix=tuning_pvalue_distributions,
            ensemble_test=ensemble_test,
            ks_pvalue_abs_threshold=ks_pvalue_abs_threshold,
            cremer_v_threshold=cremer_v_threshold,
            preferred_statistics=preferred_statistics,
        )

    if logger:
        logger.log_param("num_tests", len(reference_histogram.keys()))
        logger.log_param("Independent keys", independent_keys_group)
        if preferred_statistics:
            logger.log_param("preferred_statistics", preferred_statistics)
        logger.log_metrics(best_results)

    return ReferenceModel(
        independent_keys=list(independent_keys_group),
        reference_cdfs=reference_cdfs,
        reference_cache_suffix=cache_suffix,
    )


def run_inference_with_reference_model(
    reference_model: ReferenceModel,
    inference_dataset,
    batch_size: int,
    threshold: float,
    ensemble_test: str,
    max_workers: int,
    num_data_workers: int,
    pkl_dir: str,
    return_logits: bool,
    test_type: TestType,
    seed: int,
    logger=None,
    cache_suffix: str = "",
):
    # Inference phase (Fig. 4a): evaluate selected detectors on candidate images.
    inference_histogram = patch_parallel_preprocess(
        inference_dataset,
        batch_size,
        reference_model.independent_combinations,
        max_workers,
        num_data_workers,
        pkl_dir=pkl_dir,
        seed=seed,
        cache_suffix=cache_suffix,
    )

    inference_histogram = compute_mean_std_dict(inference_histogram)
    inference_histogram = {
        k: inference_histogram[k]
        for k in reference_model.independent_keys
        if k in inference_histogram
    }

    # Inference phase (Fig. 4b): map statistics to stored ECDFs for two-sided p-values.
    input_samples_pvalues = calculate_pvals_from_cdf(
        reference_model.reference_cdfs, inference_histogram, test_type
    )
    independent_tests_pvalues = np.array(input_samples_pvalues)
    independent_tests_pvalues = np.clip(independent_tests_pvalues, 0, 1)

    # Inference phase (Fig. 4c): aggregate independent p-values into a unified score.
    ensembled_stats, ensembled_pvalues = perform_ensemble_testing(
        independent_tests_pvalues, ensemble_test
    )
    ensembled_pvalues = np.array(ensembled_pvalues)

    if return_logits:
        return {
            "scores": 1 - ensembled_pvalues,
            "n_tests": len(list(reference_model.independent_keys)),
        }

    predictions = [1 if pval < threshold else 0 for pval in ensembled_pvalues]
    results = {
        "scores": ensembled_pvalues,
        "n_tests": len(list(reference_model.independent_keys)),
        "predictions": predictions,
    }

    if logger:
        logger.log_param("num_tests", len(reference_model.independent_keys))
        logger.log_param("Independent keys", reference_model.independent_keys)

    return results
