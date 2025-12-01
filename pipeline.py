import signal, sys, os
import argparse
from urllib.parse import urlparse
import torch.multiprocessing as mp

from statistics_factory import STATISTIC_HISTOGRAMS
mp.set_start_method("spawn", force=True)
import mlflow
from sklearn.metrics import average_precision_score, roc_curve, auc
from torchvision import transforms
from datasets_factory import DatasetFactory, DatasetType
from torch.utils.data import ConcatDataset
from stat_test import TestType, build_reference_model, run_inference_with_reference_model
from utils import set_seed, balanced_testset


def shutdown(signum, frame):
    print("Ctrl+C received, terminating workers...")
    # Terminate all active child processes
    for p in mp.active_children():
        p.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

parser = argparse.ArgumentParser(description='Statistic and Patch Testing Pipeline')
parser.add_argument('--test_type', choices=['multiple_patches', 'multiple_statistics'], default='multiple_statistics', help='Choose which type of multiple tests to perform')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for data loading.')
parser.add_argument('--sample_size', type=int, default=512, help='Sample input size after downscale.')
parser.add_argument('--patch_divisors', type=int, nargs='+', default=[0], help='Divisors to calculate patch sizes as sample_size // 2^i.')
parser.add_argument('--threshold', type=float, default=0.05, help='P-value threshold for significance testing.')
parser.add_argument('--ensemble_test', choices=['manual-stouffer', 'stouffer', 'minp'], default='manual-stouffer', help='Type of ensemble test to perform')
parser.add_argument('--dataset_type', type=str, default='ALL', choices=[e.name for e in DatasetType], help='Type of dataset configuration to use.')
parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save logs and artifacts.')
parser.add_argument('--pkls_dir', type=str, default='pkls/AIStats/new_stats', help='Path where to save pkls.')
parser.add_argument('--num_samples_per_class', type=int, default=-1, help='Number of samples per class for inference dataset.')
parser.add_argument('--num_data_workers', type=int, default=2, help='Number of workers for data loading.')
parser.add_argument('--max_workers', type=int, default=3, help='Maximum number of threads for parallel processing.')
parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')
parser.add_argument('--statistics', type=str, nargs='+', default=[k for k in STATISTIC_HISTOGRAMS if k.startswith("RIGID.") and any(k.endswith(suffix) for suffix in [".05", ".10"])])
parser.add_argument('--chi2_bins', type=int, default=10, help='Number of bins for chi-square calculations.')
parser.add_argument('--cdf_bins', type=int, default=500, help='Number of bins for cdf.')
parser.add_argument('--uniform_p_threshold', type=float, default=0.05, help='KS Threshold for uniform goodness of fit.')
parser.add_argument('--ks_pvalue_abs_threshold', type=float, default=0.4, help='Absolute KS p-value threshold for uniformity filtering.')
parser.add_argument('--cremer_v_threshold', type=float, default=0.05, help='Minimum p-value threshold for chi-square filtering.')
parser.add_argument('--preferred_statistics', type=str, nargs='*', default=["RIGID.DINO.05", "RIGID.CLIPOPENAI.05", "RIGID.DINO.10", "RIGID.CLIPOPENAI.10"], help='Statistics to prioritize when selecting the independent clique.')
parser.add_argument('--gpu', type=str, default='1', help='GPU device(s) to use, e.g., "0", "1", or "0,1".')
parser.add_argument('--run_id', type=str, default='none', help='Unique identifier for this MLflow run.')
parser.add_argument('--experiment_id', type=str, default='default', help='Name or ID of the MLflow experiment.')
parser.add_argument('--use_mlflow', type=int, default=0, help='Enable or disable MLflow logging.')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
 

def run_pipeline(logger=None):

    set_seed(args.seed)

    dataset_pkls_dir = args.pkls_dir
    os.makedirs(dataset_pkls_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((args.sample_size, args.sample_size)),
        transforms.ToTensor()
    ])

    datasets = DatasetFactory.create_dataset(dataset_type=args.dataset_type, transform=transform)
    reference_dataset = datasets['reference_real']
    test_real_dataset = datasets['test_real']
    test_fake_dataset = datasets['test_fake']

    inference_dataset = ConcatDataset([test_real_dataset, test_fake_dataset])
    labels = [0] * len(test_real_dataset) + [1] * len(test_fake_dataset)
    inference_dataset.image_paths = (
        list(getattr(test_real_dataset, "image_paths", [])) +
        list(getattr(test_fake_dataset, "image_paths", []))
    )

    patch_sizes = [args.sample_size // (2 ** d) for d in args.patch_divisors]

    test_id = f"divs_{'-'.join(map(str, args.patch_divisors))}-statistics_{len(args.statistics)}"

    if logger:
        logger.log_params(vars(args))
        logger.log_param("patch_sizes", patch_sizes)
        logger.log_param("test_id", test_id)

    reference_model = build_reference_model(
        reference_dataset=reference_dataset,
        statistics=args.statistics,
        patch_sizes=patch_sizes,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        num_data_workers=args.num_data_workers,
        pkl_dir=dataset_pkls_dir,
        seed=args.seed,
        cdf_bins=args.cdf_bins,
        ensemble_test=args.ensemble_test,
        chi2_bins=args.chi2_bins,
        ks_pvalue_abs_threshold=args.ks_pvalue_abs_threshold,
        cremer_v_threshold=args.cremer_v_threshold,
        preferred_statistics=args.preferred_statistics,
        test_type=TestType.BOTH,
        logger=logger,
    )

    results = run_inference_with_reference_model(
        reference_model=reference_model,
        inference_dataset=inference_dataset,
        batch_size=args.batch_size,
        threshold=args.threshold,
        ensemble_test=args.ensemble_test,
        max_workers=args.max_workers,
        num_data_workers=args.num_data_workers,
        pkl_dir=dataset_pkls_dir,
        return_logits=True,
        test_type=TestType.BOTH,
        logger=logger,
        seed=args.seed,
    )

    if labels is not None:
        balance_labels, balance_scores = balanced_testset(labels, results['scores'], random_state=42)
        fpr, tpr, _ = roc_curve(balance_labels, balance_scores)
        roc_auc = auc(fpr, tpr)
        ap = average_precision_score(balance_labels, balance_scores)

        if logger:
            logger.log_metric("AUC", roc_auc)
            logger.log_metric("AP", ap)


def main():

    logger = mlflow if args.use_mlflow else None

    if logger:
        mlflow.set_experiment(args.experiment_id)
        with mlflow.start_run(run_name=args.run_id):
            args.output_dir = urlparse(mlflow.get_artifact_uri()).path
            run_pipeline(logger)
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        run_pipeline(logger)


if __name__ == "__main__":
    main()
