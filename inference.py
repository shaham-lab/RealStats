import os
import sys
import argparse
from urllib.parse import urlparse
import torch.multiprocessing as mp

mp.set_start_method("spawn", force=True)
import mlflow
from sklearn.metrics import average_precision_score
from torch.utils.data import ConcatDataset
from torchvision import transforms
from datasets_factory import DatasetFactory, DatasetType
from data_utils import ImageDataset, JPEGCompressionTransform
from stat_test import TestType, build_reference_model, run_inference_with_reference_model
from utils import (
    balanced_testset,
    set_seed,
)
from transform_cache import build_transform_cache_suffix
from sklearn.metrics import roc_curve, auc


parser = argparse.ArgumentParser(description='Inference-Only Statistic and Patch Testing Pipeline')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for data loading.')
parser.add_argument('--sample_size', type=int, default=512, help='Sample input size after downscale.')
parser.add_argument('--patch_divisors', type=int, nargs='+', default=[2, 4, 8], help='Divisors to calculate patch sizes as sample_size // 2^i.')
parser.add_argument('--threshold', type=float, default=0.05, help='P-value threshold for significance testing.')
parser.add_argument('--ensemble_test', choices=['manual-stouffer', 'stouffer', 'minp'], default='manual-stouffer', help='Type of ensemble test to perform')
parser.add_argument('--dataset_type', type=str, default='ALL', choices=[e.name for e in DatasetType], help='Type of dataset to use')
parser.add_argument('--input_path', type=str, default=None, help='Optional path to a single image or a directory of images to use as the test set')
parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save logs and artifacts.')
parser.add_argument('--pkls_dir', type=str, default='pkls/AIStats/new_stats', help='Path where to save pkls.')
parser.add_argument('--num_data_workers', type=int, default=4, help='Number of workers for data loading.')
parser.add_argument('--max_workers', type=int, default=1, help='Maximum number of threads for parallel processing.')
parser.add_argument('--seed', type=int, default=38, help='Random seed for reproducibility.')
parser.add_argument('--cdf_bins', type=int, default=1000, help='Number of bins for cdf.')
parser.add_argument('--gpu', type=str, default='0', help='GPU device(s) to use, e.g., "0", "1", or "0,1".')
parser.add_argument('--run_id', type=str, default='default', help='Unique identifier for this MLflow run.')
parser.add_argument('--experiment_id', type=str, default='default', help='Name or ID of the MLflow experiment.')
parser.add_argument('--independent_keys', type=str, nargs='+', default=["PatchProcessing_statistic=RIGID.DINO.05_patch_size=512_seed=38", "PatchProcessing_statistic=RIGID.CLIPOPENAI.05_patch_size=512_seed=38"], help='Independent statistics keys group')
parser.add_argument('--inference_aug', type=str, default='none', choices=['none', 'jpeg', 'blur'], help='Apply augmentation to inference dataset (jpeg or blur).')
parser.add_argument('--latent_noise_csv', type=str, default=None, help='Path to the CSV file with LatentNoiseCriterion_original scores.')
parser.add_argument('--use_mlflow', type=int, default=0, help='Enable or disable MLflow logging.')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if args.latent_noise_csv:
    os.environ["LATENT_NOISE_CRITERION_ORIGINAL_CSV"] = os.path.abspath(args.latent_noise_csv)

    
def run_inference(logger=None):
    set_seed(args.seed)

    dataset_pkls_dir = args.pkls_dir
    os.makedirs(dataset_pkls_dir, exist_ok=True)

    base_transform_steps = [transforms.Resize((args.sample_size, args.sample_size))]
    base_transform = transforms.Compose(base_transform_steps + [transforms.ToTensor()])

    inference_transform_steps = list(base_transform_steps)
    if args.inference_aug == 'jpeg':
        inference_transform_steps.append(JPEGCompressionTransform())
    elif args.inference_aug == 'blur':
        inference_transform_steps.append(transforms.GaussianBlur(kernel_size=(3, 3), sigma=1.0))
    inference_transform = transforms.Compose(inference_transform_steps + [transforms.ToTensor()])

    reference_cache_suffix = build_transform_cache_suffix(base_transform)
    inference_cache_suffix = build_transform_cache_suffix(inference_transform)

    datasets = DatasetFactory.create_dataset(dataset_type=args.dataset_type, transform=base_transform)
    reference_dataset = datasets['reference_real']
    test_real_dataset = datasets['test_real']
    test_fake_dataset = datasets['test_fake']

    if args.input_path:
        input_path = os.path.abspath(args.input_path)
        if not (os.path.isdir(input_path) or os.path.isfile(input_path)):
            raise FileNotFoundError(f"Provided input_path does not exist: {input_path}")
        inference_dataset = ImageDataset(input_path, labels=None, transform=inference_transform)
        inference_dataset.image_paths = list(getattr(inference_dataset, 'image_paths', []))
        labels = None
    else:
        test_real_dataset.transform = inference_transform
        test_fake_dataset.transform = inference_transform
        inference_dataset = ConcatDataset([test_real_dataset, test_fake_dataset])

        real_image_paths = list(getattr(test_real_dataset, 'image_paths', []))
        fake_image_paths = list(getattr(test_fake_dataset, 'image_paths', []))
        inference_dataset.image_paths = real_image_paths + fake_image_paths

        labels = [0] * len(test_real_dataset) + [1] * len(test_fake_dataset)

    patch_sizes = [args.sample_size // (2 ** d) for d in args.patch_divisors]
    test_id = f"inference_run_{args.run_id}"

    if logger:
        logger.log_params(vars(args))
        logger.log_param("patch_sizes", patch_sizes)
        logger.log_param("test_id", test_id)
        logger.log_param("num_independent_keys", len(args.independent_keys))
        logger.log_param("reference_transform_cache_suffix", reference_cache_suffix or "none")
        logger.log_param("inference_transform_cache_suffix", inference_cache_suffix or "none")

    reference_model = build_reference_model(
        reference_dataset=reference_dataset,
        statistics=None,
        patch_sizes=None,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        num_data_workers=args.num_data_workers,
        pkl_dir=dataset_pkls_dir,
        seed=args.seed,
        cdf_bins=args.cdf_bins,
        ensemble_test=args.ensemble_test,
        chi2_bins=args.cdf_bins,
        ks_pvalue_abs_threshold=getattr(args, "uniform_p_threshold", 0.05),
        cremer_v_threshold=getattr(args, "cremer_v_threshold", 0.05),
        preferred_statistics=None,
        independent_keys=args.independent_keys,
        test_type=TestType.BOTH,
        logger=logger,
        cache_suffix=reference_cache_suffix,
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
        cache_suffix=inference_cache_suffix,
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
            run_inference(logger)
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        run_inference(logger)


if __name__ == "__main__":
    main()
