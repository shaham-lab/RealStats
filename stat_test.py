import itertools
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    calculate_metrics,
    compute_chi2_and_corr_matrix,
    compute_mean_std_dict,
    create_multiband_pvalue_grid_figure,
    create_pvalue_grid_figure,
    finding_optimal_independent_subgroup_deterministic,
    perform_ensemble_testing,
    remove_nans_from_tests,
    save_ensembled_pvalue_kde_and_images,
    save_per_image_kde_and_images,
    plot_pvalue_histograms,
    set_seed,
    plot_pvalue_histograms_from_arrays,
    save_real_statistics_kde,
    export_combined_to_csv
)
from data_utils import GlobalPatchDataset, SelfPatchDataset
from enum import Enum
from scipy.stats import kstest
import traceback
import gc


class TestType(Enum):
    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"
    

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
        
        # Determine the bin index for each sample
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


def compute_ks_pvalues(keys, pvalue_distributions, sample_size=1_000, seed=None):
    """Compute KS p-values for each statistic distribution.

    The distributions are subsampled uniformly (without replacement) to avoid
    unstable values caused by extremely low p-values.
    """

    ks_pvalues = {}
    rng = np.random.default_rng(seed)

    for key, distribution in zip(keys, pvalue_distributions):
        distribution = np.asarray(distribution)
        if distribution.size == 0:
            continue

        sample_count = min(sample_size, distribution.size)
        indices = rng.choice(distribution.size, size=sample_count, replace=False)
        sampled_distribution = distribution[indices]

        _, pvalue = kstest(sampled_distribution, "uniform")
        ks_pvalues[key] = float(pvalue)

    return ks_pvalues


def patch_parallel_preprocess(original_dataset, batch_size, combinations, max_workers, num_data_workers, pkl_dir='pkls', sort=True, seed=42, cache_suffix=""):
    """Preprocess the dataset for specific combinations in parallel."""
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


def main_multiple_patch_test(
    reference_dataset,
    inference_dataset,
    statistics,
    patch_sizes,
    test_labels=None,
    batch_size=128,
    threshold=0.05,
    save_independence_heatmaps=False,
    save_histograms=False,
    ensemble_test='stouffer',
    max_workers=128,
    num_data_workers=2,
    output_dir='logs',
    pkl_dir='pkls',
    return_logits=False,
    chi2_bins=10,
    cdf_bins=500,
    ks_pvalue_abs_threshold=0.25,
    test_type=TestType.LEFT,
    minimal_p_threshold=0.05,
    logger=None,
    seed=42,
    preferred_statistics=None
):
    """Run test for number of patches and collect sensitivity and specificity results."""
    print(f"Running test with: \npatches sizes: {patch_sizes}\nstatistics: {statistics}")

    # Generate all combinations for training
    training_combinations = generate_combinations(patch_sizes, statistics)

    # Load or compute reference histograms
    reference_histogram = patch_parallel_preprocess(
        reference_dataset, batch_size, training_combinations, max_workers, num_data_workers, pkl_dir=pkl_dir, seed=seed
    )

    reference_histogram = compute_mean_std_dict(reference_histogram)
    reference_histogram = remove_nans_from_tests(reference_histogram)

    # CDF creation
    reference_cdfs = {test_id: compute_cdf(values, bins=cdf_bins, test_id=test_id) for test_id, values in reference_histogram.items()}

    # Tuning Pvalues
    tuning_reference_pvals = calculate_pvals_from_cdf(reference_cdfs, reference_histogram, test_type)
    tuning_reference_pvals = np.clip(tuning_reference_pvals, 0, 1)

    keys = list(reference_histogram.keys())

    tuning_pvalue_distributions = tuning_reference_pvals.T

    # Chi-Square and Correlation Matrix Computation
    chi2_p_matrix, corr_matrix = compute_chi2_and_corr_matrix(
        keys, tuning_pvalue_distributions, max_workers=max_workers,
        plot_independence_heatmap=save_independence_heatmaps, output_dir=output_dir, bins=chi2_bins
    )

    independent_keys_group, best_results, optimization_roc = finding_optimal_independent_subgroup_deterministic(
        keys=keys,
        chi2_p_matrix=chi2_p_matrix,
        pvals_matrix=tuning_pvalue_distributions,
        ensemble_test=ensemble_test,
        ks_pvalue_abs_threshold=ks_pvalue_abs_threshold,
        minimal_p_threshold=minimal_p_threshold,
        preferred_statistics=preferred_statistics
    )

    if logger:
        logger.log_param("num_tests", len(reference_histogram.keys()))
        logger.log_param("Independent keys", independent_keys_group)
        if preferred_statistics:
            logger.log_param("preferred_statistics", preferred_statistics)
        logger.log_metrics(best_results)

    independent_keys_group_indices = [keys.index(value) for value in independent_keys_group]
    tuning_independent_pvals = tuning_pvalue_distributions[independent_keys_group_indices].T
    perform_ensemble_testing(tuning_independent_pvals, ensemble_test, plot=True, output_dir=output_dir)
    
    # Convert independent keys to combinations
    independent_combinations = interpret_keys_to_combinations(independent_keys_group)

    # Inference
    inference_histogram = patch_parallel_preprocess(
        inference_dataset, batch_size, independent_combinations, max_workers, num_data_workers, pkl_dir=pkl_dir, seed=seed
    )

    inference_histogram = compute_mean_std_dict(inference_histogram)

    inference_histogram = {
        k: inference_histogram[k] for k in independent_keys_group if k in inference_histogram
    }

    input_samples_pvalues = calculate_pvals_from_cdf(reference_cdfs, inference_histogram, test_type)
    independent_tests_pvalues = np.array(input_samples_pvalues)
    independent_tests_pvalues = np.clip(independent_tests_pvalues, 0, 1)

    ks_pvalues = compute_ks_pvalues(
        independent_keys_group,
        independent_tests_pvalues.T,
        sample_size=1_000,
        seed=seed,
    )

    ensembled_stats, ensembled_pvalues = perform_ensemble_testing(independent_tests_pvalues, ensemble_test)
    predictions = [1 if pval < threshold else 0 for pval in ensembled_pvalues]

    if logger:
        logger.log_dict({"ensembled_pvalues": [float(p) for p in ensembled_pvalues]}, "ensembled_pvalues.json")
        logger.log_metrics(ks_pvalues)
        logger.log_dict(ks_pvalues, "ks_pvalues_per_stat.json")

    plot_pvalue_histograms_from_arrays(
        np.array([p for p, l in zip(independent_tests_pvalues, test_labels) if l == 0]),
        np.array([p for p, l in zip(independent_tests_pvalues, test_labels) if l == 1]),
        os.path.join(output_dir, "inference_stat"),
        independent_keys_group
        )
    
    if save_histograms and test_labels:
        plot_pvalue_histograms(
            [p for p, l in zip(ensembled_pvalues, test_labels) if l == 0],
            [p for p, l in zip(ensembled_pvalues, test_labels) if l == 1],
            os.path.join(output_dir, f"histogram_plot_{patch_sizes}_{ensemble_test}_alpha_{threshold}.png"),
            "Histogram of P-values"
        )
    
    if return_logits:
        return {
            'scores': 1 - np.array(ensembled_pvalues),
            'n_tests': len(list(independent_keys_group)),
            'labels': test_labels
        }

    # Evaluate predictions
    if test_labels:
        metrics = calculate_metrics(test_labels, predictions)
        return metrics


def inference_multiple_patch_test(
    reference_dataset,
    inference_dataset,
    independent_statistics_keys_group,
    test_labels=None,
    batch_size=128,
    threshold=0.05,
    save_histograms=False,
    ensemble_test='stouffer',
    max_workers=128,
    num_data_workers=2,
    output_dir='logs',
    pkl_dir='pkls',
    return_logits=False,
    cdf_bins=500,
    test_type=TestType.LEFT,
    logger=None,
    seed=42,
    draw_pvalues_trend_figure=False,
    reference_cache_suffix="",
    cache_suffix="",
):
    """
    Simplified version of patch test for inference: no tuning, no clique finding.
    Uses all combinations and applies ensemble testing.
    """
    print(f"[INFO] Running inference-only test with statistics={independent_statistics_keys_group}")

    # Generate all combinations for training
    independent_combinations = interpret_keys_to_combinations(independent_statistics_keys_group)

    if reference_dataset is None:
        raise ValueError("reference_dataset must be provided for inference")

    real_population_histogram = patch_parallel_preprocess(
        reference_dataset, batch_size, independent_combinations, max_workers, num_data_workers, pkl_dir=pkl_dir, seed=seed, cache_suffix=reference_cache_suffix,
    )

    real_population_histogram = compute_mean_std_dict(real_population_histogram)
    real_population_histogram = remove_nans_from_tests(real_population_histogram)

    real_population_histogram = {k: v for k, v in real_population_histogram.items() if k in independent_statistics_keys_group}

    # Plot raw statistic distributions as KDE
    save_real_statistics_kde(
        real_population_histogram,
        independent_statistics_keys_group,
        output_dir,
    )
    
    # CDF creation
    real_population_cdfs = {test_id: compute_cdf(values, bins=cdf_bins, test_id=test_id) for test_id, values in real_population_histogram.items()}    

    # Tuning Pvalues
    tuning_real_population_pvals = calculate_pvals_from_cdf(real_population_cdfs, real_population_histogram, test_type)
    tuning_real_population_pvals = np.clip(tuning_real_population_pvals, 0, 1)

    keys = list(real_population_histogram.keys())
    tuning_pvalue_distributions = tuning_real_population_pvals.T

    if logger:
        logger.log_param("num_tests", len(real_population_histogram.keys()))
        logger.log_param("Independent keys", independent_statistics_keys_group)

    print(f'Independent keys: {independent_statistics_keys_group}')
    independent_keys_group_indices = [keys.index(value) for value in independent_statistics_keys_group]
    tuning_independent_pvals = tuning_pvalue_distributions[independent_keys_group_indices].T
    perform_ensemble_testing(tuning_independent_pvals, ensemble_test, plot=True, output_dir=output_dir)    
    
    # Inference
    inference_histogram = patch_parallel_preprocess(
        inference_dataset, batch_size, independent_combinations, max_workers, num_data_workers, pkl_dir=pkl_dir, seed=seed, cache_suffix=cache_suffix,
    )

    inference_histogram = compute_mean_std_dict(inference_histogram)
    inference_histogram = {
        key: inference_histogram[key] for key in independent_statistics_keys_group if key in inference_histogram
    }

    input_samples_pvalues = calculate_pvals_from_cdf(real_population_cdfs, inference_histogram, test_type)
    independent_tests_pvalues = np.array(input_samples_pvalues)
    independent_tests_pvalues = np.clip(independent_tests_pvalues, 0, 1)

    ks_pvalues = compute_ks_pvalues(
        independent_statistics_keys_group,
        independent_tests_pvalues.T,
        sample_size=1_000,
        seed=seed,
    )

    ensembled_stats, ensembled_pvalues = perform_ensemble_testing(independent_tests_pvalues, ensemble_test)
    predictions = [1 if pval < threshold else 0 for pval in ensembled_pvalues]

    if logger:
        logger.log_dict({"ensembled_pvalues": [float(p) for p in ensembled_pvalues]}, "ensembled_pvalues.json")
        logger.log_metrics(ks_pvalues)
        logger.log_dict(ks_pvalues, "ks_pvalues_per_stat.json")
    
    if test_labels and hasattr(inference_dataset, "image_paths") and draw_pvalues_trend_figure:
        combined = list(zip(inference_dataset.image_paths, ensembled_pvalues, test_labels))
        export_combined_to_csv(combined, os.path.join(output_dir, "pvalue_dist.csv"))
        for i in range(30):
            random.shuffle(combined)  # New shuffle each time

            image_paths_shuffled, pvalues_shuffled, test_labels_shuffled = zip(*combined)

            success = create_multiband_pvalue_grid_figure(
                image_paths=image_paths_shuffled,
                pvalues=pvalues_shuffled,
                test_labels=test_labels_shuffled,
                thresholds=[0.01, 0.05, 0.10, 0.25, 0.5],
                max_per_group=7,
                output_path=os.path.join(output_dir, f"significance_grid_{i}.png")
            )


    plot_pvalue_histograms_from_arrays(
        np.array([p for p, l in zip(independent_tests_pvalues, test_labels) if l == 0]),
        np.array([p for p, l in zip(independent_tests_pvalues, test_labels) if l == 1]),
        os.path.join(output_dir, "inference_stat"),
        independent_statistics_keys_group
        )
    
    if save_histograms and test_labels:
        plot_pvalue_histograms(
            [p for p, l in zip(ensembled_pvalues, test_labels) if l == 0],
            [p for p, l in zip(ensembled_pvalues, test_labels) if l == 1],
            os.path.join(output_dir, f"histogram_plot_{ensemble_test}_alpha_{threshold}.png"),
            "Histogram of P-values", 
            bins=40,
            figsize=(6, 6), title_fontsize=16, label_fontsize=14, legend_fontsize=12
        )
    
    if return_logits:
        return {
            'scores': 1 - np.array(ensembled_pvalues),
            'n_tests': len(list(independent_statistics_keys_group)),
            'labels': test_labels
        }

    # Evaluate predictions
    if test_labels:
        metrics = calculate_metrics(test_labels, predictions)
        return metrics
