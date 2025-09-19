import os
import sys
import argparse
from urllib.parse import urlparse

import mlflow
from torchvision import transforms
from datasets_factory import DatasetFactory, DatasetType
from data_utils import ImageDataset, create_inference_dataset
from stat_test import TestType, inference_multiple_patch_test_with_dependence
from utils import plot_fakeness_score_distribution, plot_fakeness_score_histogram, plot_roc_curve, set_seed

sys.setrecursionlimit(2000)

parser = argparse.ArgumentParser(description='Inference-Only Statistic and Patch Testing Pipeline')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for data loading.')
parser.add_argument('--sample_size', type=int, default=512, help='Sample input size after downscale.')
parser.add_argument('--patch_divisors', type=int, nargs='+', default=[2, 4, 8], help='Divisors to calculate patch sizes as sample_size // 2^i.')
parser.add_argument('--threshold', type=float, default=0.05, help='P-value threshold for significance testing.')
parser.add_argument('--save_histograms', type=int, choices=[0, 1], default=1, help='Save KDE plots for real and fake p-values.')
parser.add_argument('--ensemble_test', choices=['manual-stouffer', 'stouffer', 'rbm', 'minp'], default='manual-stouffer', help='Type of ensemble test to perform')
parser.add_argument('--save_independence_heatmaps', type=int, choices=[0, 1], default=1, help='Save independence test heatmaps.')
parser.add_argument('--dataset_type', type=str, default='COCO', choices=[e.name for e in DatasetType], help='Type of dataset to use')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save logs and artifacts.')
parser.add_argument('--pkls_dir', type=str, default='/data/users/haimzis/rigid_pkls', help='Path where to save pkls.')
parser.add_argument('--num_samples_per_class', type=int, default=-1, help='Number of samples per class for inference dataset.')
parser.add_argument('--num_data_workers', type=int, default=4, help='Number of workers for data loading.')
parser.add_argument('--max_workers', type=int, default=1, help='Maximum number of threads for parallel processing.')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
parser.add_argument('--chi2_bins', type=int, default=10, help='Number of bins for chi-square calculations.')
parser.add_argument('--cdf_bins', type=int, default=1000, help='Number of bins for cdf.')
parser.add_argument('--gpu', type=str, default='0', help='GPU device(s) to use, e.g., "0", "1", or "0,1".')
parser.add_argument('--run_id', type=str, required=True, help='Unique identifier for this MLflow run.')
parser.add_argument('--experiment_id', type=str, required=True, help='Name or ID of the MLflow experiment.')
parser.add_argument('--statistics_keys', type=str, nargs='+', required=True, help='Statistics keys group (all statistics to evaluate)')
parser.add_argument('--p_threshold', type=float, default=0.05, help='P-value threshold for independence graph')
parser.add_argument('--preferred_statistics', type=str, nargs='*', default=None, help='Statistics to prioritize when selecting the independent clique.')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def main():
    set_seed(args.seed)

    # Get paths dynamically based on dataset_type
    dataset_type_enum = DatasetType[args.dataset_type.upper()]
    paths = dataset_type_enum.get_paths()

    # Dynamically create a subdirectory in pkls_dir based on dataset type
    dataset_pkls_dir = os.path.join(args.pkls_dir, args.dataset_type)
    os.makedirs(dataset_pkls_dir, exist_ok=True)

    # Initialize MLflow experiment
    mlflow.set_experiment(args.experiment_id)
    with mlflow.start_run(run_name=args.run_id):
        args.output_dir = urlparse(mlflow.get_artifact_uri()).path

        # Load transforms and datasets
        transform = transforms.Compose([
            transforms.Resize((args.sample_size, args.sample_size)),
            transforms.ToTensor()
        ])

        inference_data = create_inference_dataset(paths['test_real']['path'], paths['test_fake']['path'], args.num_samples_per_class, classes='both')

        image_paths = [x[0] for x in inference_data]
        labels = [x[1] for x in inference_data]
        inference_dataset = ImageDataset(image_paths, labels, transform=transform)

        patch_sizes = [args.sample_size // (2 ** d) for d in args.patch_divisors]
        test_id = f"inference_run_{args.run_id}"

        # Log all arguments and relevant parameters
        mlflow.log_params(vars(args))
        mlflow.log_param("patch_sizes", patch_sizes)
        mlflow.log_param("test_id", test_id)
        mlflow.log_param("num_statistics_keys", len(args.statistics_keys))
        mlflow.log_param("p_threshold", args.p_threshold)

        # Run inference-only flow
        results = inference_multiple_patch_test_with_dependence(
            inference_dataset=inference_dataset,
            statistics_keys_group=args.statistics_keys,
            test_labels=labels,
            batch_size=args.batch_size,
            threshold=args.threshold,
            save_independence_heatmaps=bool(args.save_independence_heatmaps),
            save_histograms=bool(args.save_histograms),
            ensemble_test=args.ensemble_test,
            max_workers=args.max_workers,
            num_data_workers=args.num_data_workers,
            output_dir=args.output_dir,
            pkl_dir=dataset_pkls_dir,
            return_logits=True,
            chi2_bins=args.chi2_bins,
            cdf_bins=args.cdf_bins,
            test_type=TestType.BOTH,
            p_threshold=args.p_threshold,
            logger=mlflow,
            seed=args.seed,
            preferred_statistics=args.preferred_statistics
        )

        results['labels'] = labels
        auc = plot_roc_curve(results, test_id, args.output_dir)
        plot_fakeness_score_distribution(results, test_id, args.output_dir, args.threshold)
        plot_fakeness_score_histogram(results, test_id, args.output_dir, args.threshold)
        mlflow.log_metric("AUC", auc)


if __name__ == "__main__":
    main()
