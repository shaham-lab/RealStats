import signal, sys, os
import argparse
from urllib.parse import urlparse
import torch.multiprocessing as mp

from statistics_factory import STATISTIC_HISTOGRAMS
mp.set_start_method("spawn", force=True)
import mlflow
from sklearn.metrics import average_precision_score
from torchvision import transforms
from datasets_factory import DatasetFactory, DatasetType
from torch.utils.data import ConcatDataset
from stat_test import TestType, main_multiple_patch_test
from utils import plot_roc_curve, set_seed

sys.setrecursionlimit(2000)

def shutdown(signum, frame):
    print("Ctrl+C received, terminating workers...")
    # Terminate all active child processes
    for p in mp.active_children():
        p.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

# Argument parser (with only relevant arguments kept from the original script)
parser = argparse.ArgumentParser(description='Statistic and Patch Testing Pipeline')
parser.add_argument('--test_type', choices=['multiple_patches', 'multiple_statistics'], default='multiple_statistics', help='Choose which type of multiple tests to perform')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for data loading.')
parser.add_argument('--sample_size', type=int, default=512, help='Sample input size after downscale.')
parser.add_argument('--patch_divisors', type=int, nargs='+', default=[0], help='Divisors to calculate patch sizes as sample_size // 2^i.')
parser.add_argument('--threshold', type=float, default=0.05, help='P-value threshold for significance testing.')
parser.add_argument('--save_histograms', type=int, choices=[0, 1], default=1, help='Save KDE plots for real and fake p-values.')
parser.add_argument('--ensemble_test', choices=['manual-stouffer', 'stouffer', 'rbm', 'minp'], default='manual-stouffer', help='Type of ensemble test to perform')
parser.add_argument('--save_independence_heatmaps', type=int, choices=[0, 1], default=1, help='Save independence test heatmaps.')
parser.add_argument('--uniform_sanity_check', type=int, choices=[0, 1], default=0, help='Whether to perform uniform-KS sanity check.')
parser.add_argument('--dataset_type', type=str, default='MANIFOLD_BIAS_GROUP_LEAKAGE', choices=[e.name for e in DatasetType], help='Type of dataset to use (CelebA, ProGan, COCO_LEAKAGE, COCO, COCO_ALL, PROGAN_FACES_BUT_CELEBA_AS_TRAIN)')
parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save logs and artifacts.')
parser.add_argument('--pkls_dir', type=str, default='pkls/AIStats/new_stats', help='Path where to save pkls.')
parser.add_argument('--num_samples_per_class', type=int, default=-1, help='Number of samples per class for inference dataset.')
parser.add_argument('--num_data_workers', type=int, default=2, help='Number of workers for data loading.')
parser.add_argument('--max_workers', type=int, default=3, help='Maximum number of threads for parallel processing.')
parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')
parser.add_argument('--statistics', type=str, nargs='+', default=[k for k in STATISTIC_HISTOGRAMS if k.startswith("RIGID.") and any(k.endswith(suffix) for suffix in [".05", ".10"])])
parser.add_argument('--wavelet_levels', type=int, nargs='+', default=[0], help='List of wavelet levels.')
parser.add_argument('--finetune_portion', type=float, default=0.2, help='Portion of the dataset used for finetuning.')
parser.add_argument('--chi2_bins', type=int, default=10, help='Number of bins for chi-square calculations.')
parser.add_argument('--cdf_bins', type=int, default=500, help='Number of bins for cdf.')
parser.add_argument('--n_trials', type=int, default=75, help='Number of trials for optimization.')
parser.add_argument('--uniform_p_threshold', type=float, default=0.05, help='KS Threshold for uniform goodness of fit.')
parser.add_argument('--calibration_auc_threshold', type=float, default=0.5, help='Threshold for calibration AUC to filter unreliable tests.')
parser.add_argument('--ks_pvalue_abs_threshold', type=float, default=0.4, help='Absolute KS p-value threshold for uniformity filtering.')
parser.add_argument('--minimal_p_threshold', type=float, default=0.05, help='Minimum p-value threshold for chi-square filtering.')
parser.add_argument('--preferred_statistics', type=str, nargs='*', default=None, help='Statistics to prioritize when selecting the independent clique.')
parser.add_argument('--gpu', type=str, default='1', help='GPU device(s) to use, e.g., "0", "1", or "0,1".')
parser.add_argument('--run_id', type=str, default='none', help='Unique identifier for this MLflow run.')
parser.add_argument('--experiment_id', type=str, default='default', help='Name or ID of the MLflow experiment.')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
 

def main():

    set_seed(args.seed)

    # Get paths dynamically based on dataset_type
    dataset_type_enum = DatasetType[args.dataset_type.upper()]
    paths = dataset_type_enum.get_paths()

    # Dynamically create a subdirectory in pkls_dir based on dataset type
    # dataset_pkls_dir = os.path.join(args.pkls_dir, args.dataset_type)
    # os.makedirs(dataset_pkls_dir, exist_ok=True)
    dataset_pkls_dir = args.pkls_dir
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

        datasets = DatasetFactory.create_dataset(dataset_type=args.dataset_type, transform=transform)
        reference_dataset = datasets['reference_real']
        test_real_dataset = datasets['test_real']
        test_fake_dataset = datasets['test_fake']

        # Instead of create_inference_dataset, just combine them
        inference_dataset = ConcatDataset([test_real_dataset, test_fake_dataset])
        labels = [0] * len(test_real_dataset) + [1] * len(test_fake_dataset) # TODO: this is not feasible # [sample[1] for sample in inference_dataset]
        inference_dataset.image_paths = (
            list(getattr(test_real_dataset, "image_paths", [])) +
            list(getattr(test_fake_dataset, "image_paths", []))
        )

        # Compute patch sizes from divisors
        patch_sizes = [args.sample_size // (2 ** d) for d in args.patch_divisors]

        test_id = f"divs_{'-'.join(map(str, args.patch_divisors))}-statistics_{len(args.statistics)}"

        # Log all relevant parameters
        mlflow.log_params(vars(args))
        mlflow.log_param("patch_sizes", patch_sizes)
        mlflow.log_param("test_id", test_id)


        results = main_multiple_patch_test(
            reference_dataset=reference_dataset,
            inference_dataset=inference_dataset,
            test_labels=labels,
            batch_size=args.batch_size,
            threshold=args.threshold,
            patch_sizes=patch_sizes,
            statistics=args.statistics,
            wavelet_levels=args.wavelet_levels,
            save_independence_heatmaps=bool(args.save_independence_heatmaps),
            save_histograms=bool(args.save_histograms),
            ensemble_test=args.ensemble_test,
            max_workers=args.max_workers,
            num_data_workers=args.num_data_workers,
            output_dir=args.output_dir,
            pkl_dir=dataset_pkls_dir,
            return_logits=True,
            portion=args.finetune_portion,
            chi2_bins=args.chi2_bins,
            cdf_bins=args.cdf_bins,
            n_trials=args.n_trials,
            uniform_p_threshold=args.uniform_p_threshold,
            uniform_sanity_check=bool(args.uniform_sanity_check),
            calibration_auc_threshold=args.calibration_auc_threshold,
            ks_pvalue_abs_threshold=args.ks_pvalue_abs_threshold,
            minimal_p_threshold=args.minimal_p_threshold,
            test_type=TestType.BOTH,
            logger=mlflow,
            seed=args.seed,
            preferred_statistics=args.preferred_statistics
        )

        results['labels'] = labels
        auc = plot_roc_curve(results, test_id, args.output_dir)
        ap = average_precision_score(labels, results['scores'])
        mlflow.log_metric("AUC", auc)
        mlflow.log_metric("AP", ap)


if __name__ == "__main__":
    main()
