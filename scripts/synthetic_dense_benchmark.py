import argparse
from urllib.parse import urlparse

import mlflow

from utils.utils import (
    compute_chi2_and_corr_matrix,
    find_largest_independent_group_iterative,
    synthesize_dense_statistics,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Synthetic dense independence benchmark")
    parser.add_argument("--num_statistics", type=int, default=32, help="Number of synthetic statistics to generate.")
    parser.add_argument("--vector_length", type=int, default=10_000, help="Length of each statistic vector.")
    parser.add_argument("--chi2_bins", type=int, default=10, help="Number of bins for chi-square calculations.")
    parser.add_argument("--independence_threshold", type=float, default=0.05, help="Threshold for clique construction.")
    parser.add_argument("--correlation_strength", type=float, default=0.9, help="Shared signal weight controlling graph density.")
    parser.add_argument("--noise_scale", type=float, default=0.05, help="Noise level added per statistic.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--experiment_id", type=str, default="synthetic-benchmark", help="MLflow experiment name or ID.")
    parser.add_argument("--run_name", type=str, default="dense-benchmark", help="MLflow run name.")
    return parser.parse_args()


def main():
    args = parse_args()

    mlflow.set_experiment(args.experiment_id)
    with mlflow.start_run(run_name=args.run_name):
        args.output_dir = urlparse(mlflow.get_artifact_uri()).path

        mlflow.log_params({
            "num_statistics": args.num_statistics,
            "vector_length": args.vector_length,
            "chi2_bins": args.chi2_bins,
            "independence_threshold": args.independence_threshold,
            "correlation_strength": args.correlation_strength,
            "noise_scale": args.noise_scale,
            "seed": args.seed,
        })

        keys, distributions = synthesize_dense_statistics(
            args.num_statistics,
            args.vector_length,
            correlation_strength=args.correlation_strength,
            noise_scale=args.noise_scale,
            seed=args.seed,
        )

        chi2_p_matrix, _ = compute_chi2_and_corr_matrix(
            keys,
            distributions,
            plot_independence_heatmap=False,
            bins=args.chi2_bins,
            logger=mlflow,
        )

        cliques = find_largest_independent_group_iterative(
            keys,
            chi2_p_matrix,
            p_threshold=args.independence_threshold,
            test_type="cramer_v",
            logger=mlflow,
        )

        pairwise_comparisons = args.num_statistics * (args.num_statistics - 1) // 2
        worst_case_operations = pairwise_comparisons * args.vector_length
        mlflow.log_metric("theoretical_pairwise_comparisons", pairwise_comparisons)
        mlflow.log_metric("worst_case_value_ops", worst_case_operations)
        mlflow.log_metric("num_maximal_cliques", len(cliques))

        print(f"Finished synthetic benchmark with {len(keys)} statistics and {args.vector_length} samples per statistic")
        print(f"Found {len(cliques)} maximal cliques at threshold={args.independence_threshold}")


if __name__ == "__main__":
    main()
