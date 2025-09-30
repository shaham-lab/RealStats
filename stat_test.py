import itertools
import math
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import random
import re
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from statistics_factory import get_histogram_generator
from processing.manifold_bias_histogram import PathBasedStatistic
from utils import (
    AUC_tests_filter,
    compute_cdf,
    calculate_metrics,
    compute_chi2_and_corr_matrix,
    compute_mean_std_dict,
    create_multiband_pvalue_grid_figure,
    create_pvalue_grid_figure,
    find_largest_independent_group,
    find_largest_independent_group_with_plot,
    find_largest_independent_group_iterative,
    find_largest_uncorrelated_group,
    finding_optimal_independent_subgroup,
    finding_optimal_uncorrelated_subgroup,
    finding_optimal_independent_subgroup_deterministic,
    ks_uniform_sanity_check,
    perform_ensemble_testing,
    get_total_size_in_MB,
    plot_ks_vs_pthreshold,
    plot_stouffer_analysis,
    plot_uniform_and_nonuniform,
    remove_nans_from_tests,
    save_ensembled_pvalue_kde_and_images,
    save_per_image_kde_and_images,
    plot_pvalue_histograms,
    plot_binned_histogram,
    plot_histograms,
    save_to_csv,
    set_seed,
    split_population_histogram,
    plot_cdf,
    compute_dist_cdf,
    plot_pvalue_histograms_from_arrays,
    save_real_statistics_kde
)
from statsmodels.stats.multitest import multipletests
from data_utils import GlobalPatchDataset, SelfPatchDataset
from enum import Enum
import time
import gc
from utils.utils import compute_mi_and_corr_matrix


class DataType(Enum):
    TRAIN = "train"
    CALIB = "calib"
    TEST = "test"
    TUNING = "tuning"


class TestType(Enum):
    LEFT = "left"
    RIGHT = "right"
    BOTH = "both"
    

def get_unique_id(patch_size, level, statistic, seed=42):
    """Generate a unique ID for processing."""
    return f"PatchProcessing_statistic={statistic}_level={level}_patch_size={patch_size}_seed={seed}"


def preprocess_statistic(dataset, batch_size, statistic, level, num_data_workers, patch_size, pkl_dir, data_type: DataType, seed=42, cache_suffix=""):
    """Preprocess the dataset for a single statistic name and level using various histogram statistics."""
    set_seed(seed)

    unique_id = get_unique_id(patch_size, level, statistic, seed)
    combo_dir_name = f"{unique_id}{cache_suffix}" if cache_suffix else unique_id
    combo_dir = os.path.join(pkl_dir, combo_dir_name)
    results = []

    expected_stat_paths = [
        os.path.join(combo_dir, os.path.splitext(os.path.abspath(p).lstrip(os.sep))[0] + ".npy")
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

    histogram_generator = get_histogram_generator(statistic, level)
    if histogram_generator is None:
        return None

    is_path_based = isinstance(histogram_generator, PathBasedStatistic)

    for images, _, paths in tqdm(data_loader, desc=f"Processing {statistic}-{level}", unit="batch", total=len(data_loader), leave=False):
        B, P = images.shape[:2]

        cached = [None] * B
        to_compute = []

        for i, path in enumerate(paths):
            stat_path = os.path.join(combo_dir, os.path.splitext(os.path.abspath(path).lstrip(os.sep))[0] + ".npy")
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


def fdr_classification(pvalues, threshold=0.05):
    """Perform FDR correction and return whether any p-values are significant."""
    _, pvals_corrected, _, _ = multipletests(pvalues, method='fdr_bh', alpha=threshold)
    return int(np.any(pvals_corrected < threshold))


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


def calculate_pvals_from_cdf(real_population_cdfs, samples_histogram, split="Input's", test_type=TestType.LEFT):
    input_samples = [dict(zip(samples_histogram.keys(), values)) for values in zip(*samples_histogram.values())]
    validate_real_population_cdfs(real_population_cdfs, input_samples)
    input_samples_pvalues = [calculate_cdfs_pvalues(real_population_cdfs, sample, test_type) for sample in tqdm(input_samples, desc=f"Calculating {split} P-values")]
    return input_samples_pvalues


def generate_combinations(patch_sizes, statistics, wavelet_levels):
    """Generate all combinations for the training phase."""
    combinations = []
    for patch_size, statistic, level in itertools.product(patch_sizes, statistics, wavelet_levels):
        combinations.append({
            'patch_size': patch_size,
            'statistic': statistic,
            'level': level,
        })
    return combinations


def interpret_keys_to_combinations(independent_keys_group):
    """Convert independent keys to relevant test combinations and preserve order while removing duplicates."""
    seen = set()
    combinations = []

    for key in independent_keys_group:
        match = re.match(r"PatchProcessing_statistic=([\w.]+)_level=(\d+)_patch_size=(\d+)", key)
        if match:
            statistic, level, patch_size = match.groups()
            combination = {
                'statistic': statistic,
                'level': int(level),
                'patch_size': int(patch_size),
            }
            key_tuple = (statistic, int(level), int(patch_size))
            if key_tuple not in seen:
                seen.add(key_tuple)
                combinations.append(combination)

    return combinations


def patch_parallel_preprocess(original_dataset, batch_size, combinations, max_workers, num_data_workers, pkl_dir='pkls', data_type: DataType = DataType.TRAIN, sort=True, seed=42, cache_suffix=""):
    """Preprocess the dataset for specific combinations in parallel."""
    results = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_combination = {
            executor.submit(preprocess_statistic, SelfPatchDataset(original_dataset, comb['patch_size']), batch_size, comb['statistic'], comb['level'], num_data_workers, comb['patch_size'], pkl_dir, data_type, seed, cache_suffix): comb
            for comb in combinations
        }

        for future in tqdm(as_completed(future_to_combination), total=len(future_to_combination), desc="Processing..."):
            combination = future_to_combination[future]
            unique_id = get_unique_id(combination['patch_size'], combination['level'], combination['statistic'], seed)
            try:
                results[unique_id] = future.result()
            except Exception as exc:
                print(f"Combination {combination}, generated an exception: {exc}")

        results = {k: v for k, v in results.items() if v is not None}
        # results = {k.replace(f"_{data_type.value}", ""): v for k, v in results.items()} # TODO: Check if datatype is needed

        if sort:
            results = {k: results[k] for k in sorted(results)}

    return results


def main_multiple_patch_test(
    reference_dataset,
    inference_dataset,
    statistics,
    wavelet_levels,
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
    portion=0.2,
    chi2_bins=10,
    cdf_bins=500,
    n_trials=100,
    uniform_p_threshold=0.05,
    calibration_auc_threshold=0.3,
    uniform_sanity_check=False,
    ks_pvalue_abs_threshold=0.25,
    test_type=TestType.LEFT,
    minimal_p_threshold=0.05,
    logger=None,
    seed=42,
    preferred_statistics=None
):
    """Run test for number of patches and collect sensitivity and specificity results."""
    print(f"Running test with: \npatches sizes: {patch_sizes}\nstatistics: {statistics}\nlevels: {wavelet_levels}")

    # Generate all combinations for training
    training_combinations = generate_combinations(patch_sizes, statistics, wavelet_levels)

    # Load or compute reference histograms
    reference_histogram = patch_parallel_preprocess(
        reference_dataset, batch_size, training_combinations, max_workers, num_data_workers, pkl_dir=pkl_dir, data_type=DataType.TRAIN, seed=seed
    )

    reference_histogram = compute_mean_std_dict(reference_histogram)
    reference_histogram = remove_nans_from_tests(reference_histogram)

    # Spliting 
    # tuning_histogram, training_histogram = split_population_histogram(reference_histogram, portion)
    tuning_histogram, training_histogram = reference_histogram, reference_histogram

    # CDF creation
    reference_cdfs = {test_id: compute_cdf(values, bins=cdf_bins, test_id=test_id) for test_id, values in training_histogram.items()}

    # Tuning Pvalues
    tuning_reference_pvals = calculate_pvals_from_cdf(reference_cdfs, tuning_histogram, DataType.TUNING.name, test_type)
    tuning_reference_pvals = np.clip(tuning_reference_pvals, 0, 1)

    keys = list(reference_histogram.keys())

    tuning_pvalue_distributions = tuning_reference_pvals.T
 
    if uniform_sanity_check:
        # Ignore non uniform distributions
        keys, tuning_reference_pvals = ks_uniform_sanity_check(
            output_dir, uniform_p_threshold, logger, tuning_reference_pvals, tuning_pvalue_distributions, keys
        )

    # Chi-Square and Correlation Matrix Computation
    chi2_p_matrix, corr_matrix = compute_chi2_and_corr_matrix(
        keys, tuning_pvalue_distributions, max_workers=max_workers,
        plot_independence_heatmap=save_independence_heatmaps, output_dir=output_dir, bins=chi2_bins
    )
    
    # chi2_p_matrix, corr_matrix = compute_mi_and_corr_matrix(
    #     keys, tuning_pvalue_distributions, max_workers=max_workers,
    #     plot_independence_heatmap=save_independence_heatmaps, output_dir=output_dir, bins=chi2_bins
    # )

    largest_independent_clique_size_approximation = len(find_largest_independent_group(keys, chi2_p_matrix, 0.05, "cramer_v"))

    independent_keys_group, best_results, optimization_roc = finding_optimal_independent_subgroup_deterministic(
        keys=keys,
        chi2_p_matrix=chi2_p_matrix,
        pvals_matrix=tuning_pvalue_distributions,
        ensemble_test=ensemble_test,
        fake_pvals_matrix=None,
        ks_pvalue_abs_threshold=ks_pvalue_abs_threshold,
        minimal_p_threshold=minimal_p_threshold,
        preferred_statistics=preferred_statistics
    )

    print(f'Relexation largest clique approximation: {largest_independent_clique_size_approximation}')

    if logger:
        logger.log_param("num_tests", len(reference_histogram.keys()))
        logger.log_param("Independent keys", independent_keys_group)
        if preferred_statistics:
            logger.log_param("preferred_statistics", preferred_statistics)
        logger.log_metric("largest_independent_clique_size_approximation", largest_independent_clique_size_approximation)
        logger.log_metrics(best_results)

    independent_keys_group_indices = [keys.index(value) for value in independent_keys_group]
    tuning_independent_pvals = tuning_pvalue_distributions[independent_keys_group_indices].T
    perform_ensemble_testing(tuning_independent_pvals, ensemble_test, plot=True, output_dir=output_dir)
    
    # Convert independent keys to combinations
    independent_combinations = interpret_keys_to_combinations(independent_keys_group)

    # Inference
    inference_histogram = patch_parallel_preprocess(
        inference_dataset, batch_size, independent_combinations, max_workers, num_data_workers, pkl_dir=pkl_dir, data_type=DataType.TEST, seed=seed
    )

    inference_histogram = compute_mean_std_dict(inference_histogram)

    inference_histogram = {
        k: inference_histogram[k] for k in independent_keys_group if k in inference_histogram
    }

    input_samples_pvalues = calculate_pvals_from_cdf(reference_cdfs, inference_histogram, DataType.TEST.name, test_type)
    independent_tests_pvalues = np.array(input_samples_pvalues)
    independent_tests_pvalues = np.clip(independent_tests_pvalues, 0, 1)
        
    ensembled_stats, ensembled_pvalues = perform_ensemble_testing(independent_tests_pvalues, ensemble_test)
    predictions = [1 if pval < threshold else 0 for pval in ensembled_pvalues]

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
            'n_tests': len(list(independent_keys_group))
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
        reference_dataset,
        batch_size,
        independent_combinations,
        max_workers,
        num_data_workers,
        pkl_dir=pkl_dir,
        data_type=DataType.TRAIN,
        seed=seed,
        cache_suffix=reference_cache_suffix,
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

    # Spliting 
    # tuning_histogram, training_histogram = split_population_histogram(real_population_histogram, portion)
    tuning_histogram, training_histogram = real_population_histogram, real_population_histogram
    
    # CDF creation
    real_population_cdfs = {test_id: compute_cdf(values, bins=cdf_bins, test_id=test_id) for test_id, values in training_histogram.items()}    

    # Tuning Pvalues
    tuning_real_population_pvals = calculate_pvals_from_cdf(real_population_cdfs, tuning_histogram, DataType.TUNING.name, test_type)
    tuning_real_population_pvals = np.clip(tuning_real_population_pvals, 0, 1)

    keys = list(real_population_histogram.keys())
    keys = [k.replace(f"_{DataType.TRAIN}", "") for k in keys]

    tuning_pvalue_distributions = tuning_real_population_pvals.T

    # Chi-Square and Correlation Matrix Computation
    chi2_p_matrix, corr_matrix = compute_chi2_and_corr_matrix(
        keys, tuning_pvalue_distributions, max_workers=max_workers,
        plot_independence_heatmap=save_independence_heatmaps, output_dir=output_dir, bins=chi2_bins
    )
    
    largest_independent_clique_size_approximation = len(find_largest_independent_group(keys, chi2_p_matrix, 0.05))
    
    print(f'Relexation largest clique approximation: {largest_independent_clique_size_approximation}')

    if logger:
        logger.log_param("num_tests", len(real_population_histogram.keys()))
        logger.log_param("Independent keys", independent_statistics_keys_group)
        logger.log_metric("largest_independent_clique_size_approximation", largest_independent_clique_size_approximation)

    independent_keys_group_indices = [keys.index(value) for value in independent_statistics_keys_group]
    tuning_independent_pvals = tuning_pvalue_distributions[independent_keys_group_indices].T
    _, tuning_ensembled_pvalues = perform_ensemble_testing(tuning_independent_pvals, ensemble_test, plot=True, output_dir=output_dir)    
    
    # Inference
    inference_histogram = patch_parallel_preprocess(
        inference_dataset,
        batch_size,
        independent_combinations,
        max_workers,
        num_data_workers,
        pkl_dir=pkl_dir,
        data_type=DataType.TEST,
        seed=seed,
        cache_suffix=cache_suffix,
    )

    inference_histogram = compute_mean_std_dict(inference_histogram)
    inference_histogram = {
        key: inference_histogram[key] for key in independent_statistics_keys_group if key in inference_histogram
    }

    input_samples_pvalues = calculate_pvals_from_cdf(real_population_cdfs, inference_histogram, DataType.TEST.name, test_type)
    independent_tests_pvalues = np.array(input_samples_pvalues)
    independent_tests_pvalues = np.clip(independent_tests_pvalues, 0, 1)

    # # Save per-image KDE plots and images
    # if test_labels and hasattr(inference_dataset, "image_paths"):
    #     save_per_image_kde_and_images(
    #         image_paths=inference_dataset.image_paths,
    #         test_labels=test_labels,
    #         tuning_real_population_pvals=tuning_independent_pvals,
    #         input_samples_pvalues=input_samples_pvalues,
    #         independent_statistics_keys_group=independent_statistics_keys_group,
    #         output_dir=output_dir,
    #         max_per_class=10
    #     )
    
    ensembled_stats, ensembled_pvalues = perform_ensemble_testing(independent_tests_pvalues, ensemble_test)
    predictions = [1 if pval < threshold else 0 for pval in ensembled_pvalues]

    # if test_labels and hasattr(inference_dataset, "image_paths"):
    #     save_ensembled_pvalue_kde_and_images(
    #         image_paths=inference_dataset.image_paths,
    #         test_labels=test_labels,
    #         ensembled_pvalues=ensembled_pvalues,
    #         tuning_ensembled_pvalues=tuning_ensembled_pvalues,
    #         output_dir=output_dir,
    #         max_per_class=10
    #     )
    
    if test_labels and hasattr(inference_dataset, "image_paths") and draw_pvalues_trend_figure:
        combined = list(zip(inference_dataset.image_paths, ensembled_pvalues, test_labels))

        for i in range(30):
            random.shuffle(combined)  # New shuffle each time

            image_paths_shuffled, pvalues_shuffled, test_labels_shuffled = zip(*combined)

            success = create_multiband_pvalue_grid_figure(
                image_paths=image_paths_shuffled,
                pvalues=pvalues_shuffled,
                test_labels=test_labels_shuffled,
                thresholds=[0.01, 0.05, 0.10, 0.25, 0.5],
                max_per_group=6,
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
            bins=50,
            figsize=(6, 6), title_fontsize=16, label_fontsize=14, legend_fontsize=12
        )
    
    if return_logits:
        return {
            'scores': 1 - np.array(ensembled_pvalues),
            'n_tests': len(list(independent_statistics_keys_group))
        }

    # Evaluate predictions
    if test_labels:
        metrics = calculate_metrics(test_labels, predictions)
        return metrics



def inference_multiple_patch_test_with_dependence(
    reference_dataset,
    inference_dataset,
    statistics_keys_group,
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
    p_threshold=0.05,
    test_type=TestType.LEFT,
    logger=None,
    seed=42,
    preferred_statistics=None,
    reference_cache_suffix="",
    cache_suffix="",
):
    """Inference pipeline that automatically finds an independent subset using max clique."""
    print(f"[INFO] Running inference with dependence analysis: {statistics_keys_group}")

    if reference_dataset is None:
        raise ValueError("reference_dataset must be provided for inference")

    all_combinations = interpret_keys_to_combinations(statistics_keys_group)

    real_population_histogram = patch_parallel_preprocess(
        reference_dataset,
        batch_size,
        all_combinations,
        max_workers,
        num_data_workers,
        pkl_dir=pkl_dir,
        data_type=DataType.TRAIN,
        seed=seed,
        cache_suffix=reference_cache_suffix,
    )

    real_population_histogram = compute_mean_std_dict(real_population_histogram)
    real_population_histogram = remove_nans_from_tests(real_population_histogram)

    real_population_histogram = {k: v for k, v in real_population_histogram.items() if k in statistics_keys_group}

    # Plot raw statistic distributions as KDE
    save_real_statistics_kde(
        real_population_histogram,
        statistics_keys_group,
        output_dir,
    )

    tuning_histogram, training_histogram = real_population_histogram, real_population_histogram

    real_population_cdfs = {test_id: compute_cdf(values, bins=cdf_bins, test_id=test_id) for test_id, values in training_histogram.items()}
    tuning_real_population_pvals = calculate_pvals_from_cdf(real_population_cdfs, tuning_histogram, DataType.TUNING.name, test_type)
    tuning_real_population_pvals = np.clip(tuning_real_population_pvals, 0, 1)

    keys = list(real_population_histogram.keys())
    keys = [k.replace(f"_{DataType.TRAIN}", "") for k in keys]

    tuning_pvalue_distributions = tuning_real_population_pvals.T

    start_time = time.time()
    chi2_p_matrix, _ = compute_chi2_and_corr_matrix(
        keys, tuning_pvalue_distributions, max_workers=max_workers,
        plot_independence_heatmap=save_independence_heatmaps, output_dir=output_dir, bins=chi2_bins
    )

    chi2_duration = (time.time() - start_time) * 1000  # in ms

    initial_independent_group = find_largest_independent_group_with_plot(keys, chi2_p_matrix, p_threshold, output_dir)

    start_time = time.time()
    independent_keys_group, best_results, _ = finding_optimal_independent_subgroup_deterministic(
        keys=keys,
        chi2_p_matrix=chi2_p_matrix,
        pvals_matrix=tuning_pvalue_distributions,
        ensemble_test=ensemble_test,
        fake_pvals_matrix=None,
        ks_pvalue_abs_threshold=0.5,
        minimal_p_threshold=0.05,
        preferred_statistics=preferred_statistics
    )
    clique_duration = (time.time() - start_time) * 1000  # in ms

    # After real_population_histogram is ready
    hist_size_mb = get_total_size_in_MB(real_population_histogram)
    cdf_size_mb = get_total_size_in_MB(real_population_cdfs)
    
    if logger:
        logger.log_param("num_tests", len(real_population_histogram.keys()))
        logger.log_param("Independent keys", independent_keys_group)
        if preferred_statistics:
            logger.log_param("preferred_statistics", preferred_statistics)
        logger.log_param("initial_clique_estimate", initial_independent_group)
        logger.log_metric("graph_reconstruction_and_max_clique_timer", clique_duration)
        logger.log_metric("chi2_pair_wise_timer", chi2_duration)
        logger.log_metric("real_population_histogram_MB", round(hist_size_mb, 2))
        logger.log_metric("real_population_cdfs_MB", round(cdf_size_mb, 2))
        if best_results:
            logger.log_metrics(best_results)

    independent_indices = [keys.index(value) for value in independent_keys_group]
    tuning_independent_pvals = tuning_pvalue_distributions[independent_indices].T
    _, tuning_ensembled_pvalues = perform_ensemble_testing(tuning_independent_pvals, ensemble_test, plot=True, output_dir=output_dir)

    # Compute histograms for all statistics
    all_combinations = interpret_keys_to_combinations(statistics_keys_group)

    start_stat_extraction = time.time()
    all_inference_histogram = patch_parallel_preprocess(inference_dataset, batch_size, all_combinations, max_workers, num_data_workers, pkl_dir=pkl_dir, data_type=DataType.TEST, seed=seed, cache_suffix=cache_suffix)
    end_stat_extraction = time.time()
    elapsed_stat_extraction = (end_stat_extraction - start_stat_extraction) * 1000  # ms

    if logger:
        logger.log_metric("statistic_extraction_timer", round(elapsed_stat_extraction, 2))

    all_inference_histogram = compute_mean_std_dict(all_inference_histogram)

    # P-values for all statistics
    input_all_samples_pvalues = calculate_pvals_from_cdf(real_population_cdfs, all_inference_histogram, DataType.TEST.name, test_type)
    all_tests_pvalues = np.clip(np.array(input_all_samples_pvalues), 0, 1)

    # Focus on the independent subset for inference
    inference_histogram = {key: all_inference_histogram[key] for key in independent_keys_group if key in all_inference_histogram}

    input_samples_pvalues = calculate_pvals_from_cdf(real_population_cdfs, inference_histogram, DataType.TEST.name, test_type)
    independent_tests_pvalues = np.clip(np.array(input_samples_pvalues), 0, 1)

    if test_labels and hasattr(inference_dataset, "image_paths"):
        save_per_image_kde_and_images(
            image_paths=inference_dataset.image_paths,
            test_labels=test_labels,
            tuning_real_population_pvals=tuning_independent_pvals,
            input_samples_pvalues=input_samples_pvalues,
            independent_statistics_keys_group=independent_keys_group,
            output_dir=output_dir,
            max_per_class=10
        )

    ensembled_stats, ensembled_pvalues = perform_ensemble_testing(independent_tests_pvalues, ensemble_test)
    predictions = [1 if pval < threshold else 0 for pval in ensembled_pvalues]

    if test_labels and hasattr(inference_dataset, "image_paths"):
        save_ensembled_pvalue_kde_and_images(
            image_paths=inference_dataset.image_paths,
            test_labels=test_labels,
            ensembled_pvalues=ensembled_pvalues,
            tuning_ensembled_pvalues=tuning_ensembled_pvalues,
            output_dir=output_dir,
            max_per_class=10
        )

    # Plot per-statistic histograms for all tests
    plot_pvalue_histograms_from_arrays(
        np.array([p for p, l in zip(all_tests_pvalues, test_labels) if l == 0]),
        np.array([p for p, l in zip(all_tests_pvalues, test_labels) if l == 1]),
        os.path.join(output_dir, "all_statistics"),
        statistics_keys_group
    )

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
            os.path.join(output_dir, f"histogram_plot_{ensemble_test}_alpha_{threshold}.png"),
            "Histogram of P-values",
        )

    if return_logits:
        return {
            'scores': 1 - np.array(ensembled_pvalues),
            'n_tests': len(list(independent_keys_group))
        }

    if test_labels:
        metrics = calculate_metrics(test_labels, predictions)
        return metrics