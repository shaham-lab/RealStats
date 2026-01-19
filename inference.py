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
from data_utils import JPEGCompressionTransform
from stat_test import TestType, inference_multiple_patch_test
from utils import (
    balanced_testset,
    plot_fakeness_score_histogram,
    plot_roc_curve,
    set_seed,
    export_score_bins_table,
)
from utils.transform_cache import build_transform_cache_suffix
from torchvision.utils import save_image


parser = argparse.ArgumentParser(description='Inference-Only Statistic and Patch Testing Pipeline')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for data loading.')
parser.add_argument('--sample_size', type=int, default=512, help='Sample input size after downscale.')
parser.add_argument('--patch_divisors', type=int, nargs='+', default=[2, 4, 8], help='Divisors to calculate patch sizes as sample_size // 2^i.')
parser.add_argument('--threshold', type=float, default=0.05, help='P-value threshold for significance testing.')
parser.add_argument('--save_histograms', type=int, choices=[0, 1], default=1, help='Save KDE plots for real and fake p-values.')
parser.add_argument('--ensemble_test', choices=['manual-stouffer', 'stouffer', 'rbm', 'minp'], default='manual-stouffer', help='Type of ensemble test to perform')
parser.add_argument('--dataset_type', type=str, default='ALL', choices=[e.name for e in DatasetType], help='Type of dataset to use')
parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save logs and artifacts.')
parser.add_argument('--pkls_dir', type=str, default='pkls/AIStats/new_stats', help='Path where to save pkls.')
parser.add_argument('--num_data_workers', type=int, default=6, help='Number of workers for data loading.')
parser.add_argument('--max_workers', type=int, default=1, help='Maximum number of threads for parallel processing.')
parser.add_argument('--seed', type=int, default=38, help='Random seed for reproducibility.')
parser.add_argument('--cdf_bins', type=int, default=1000, help='Number of bins for cdf.')
parser.add_argument('--gpu', type=str, default='0', help='GPU device(s) to use, e.g., "0", "1", or "0,1".')
parser.add_argument('--run_id', type=str, default='default', help='Unique identifier for this MLflow run.')
parser.add_argument('--experiment_id', type=str, default='default', help='Name or ID of the MLflow experiment.')
parser.add_argument('--independent_keys', type=str, nargs='+', default=["PatchProcessing_statistic=RIGID.DINO.05_patch_size=512_seed=38", "PatchProcessing_statistic=RIGID.CLIPOPENAI.05_patch_size=512_seed=38"], help='Independent statistics keys group')
parser.add_argument('--inference_aug', type=str, default='none', choices=['none', 'jpeg', 'blur'], help='Apply augmentation to inference dataset (jpeg or blur).')
parser.add_argument('--draw_pvalues_trend_figure', type=int, default=0, help='whether to draw p-values trend figure')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    
def main():
    set_seed(args.seed)

    dataset_pkls_dir = args.pkls_dir
    os.makedirs(dataset_pkls_dir, exist_ok=True)

    # Initialize MLflow experiment
    mlflow.set_experiment(args.experiment_id)
    with mlflow.start_run(run_name=args.run_id):
        args.output_dir = urlparse(mlflow.get_artifact_uri()).path

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

        # Apply the inference-time augmentation only to the evaluation splits.
        test_real_dataset.transform = inference_transform
        test_fake_dataset.transform = inference_transform

        inference_dataset = ConcatDataset([test_real_dataset, test_fake_dataset])

        real_image_paths = list(getattr(test_real_dataset, 'image_paths', []))
        fake_image_paths = list(getattr(test_fake_dataset, 'image_paths', []))
        inference_dataset.image_paths = real_image_paths + fake_image_paths

        labels = [0] * len(test_real_dataset) + [1] * len(test_fake_dataset)

        first_sample = inference_dataset[0]
        if isinstance(first_sample, tuple) and len(first_sample) == 3:
            img_tensor, label, _ = first_sample
        else:
            img_tensor, label = first_sample
        save_image(img_tensor, os.path.join(args.output_dir, "example_image.png"))

        patch_sizes = [args.sample_size // (2 ** d) for d in args.patch_divisors]
        test_id = f"inference_run_{args.run_id}"

        # Log all arguments and relevant parameters
        mlflow.log_params(vars(args))
        mlflow.log_param("patch_sizes", patch_sizes)
        mlflow.log_param("test_id", test_id)
        mlflow.log_param("num_independent_keys", len(args.independent_keys))
        mlflow.log_param("reference_transform_cache_suffix", reference_cache_suffix or "none")
        mlflow.log_param("inference_transform_cache_suffix", inference_cache_suffix or "none")

        # Run inference-only flow
        results = inference_multiple_patch_test(
            reference_dataset=reference_dataset,
            inference_dataset=inference_dataset,
            independent_statistics_keys_group=args.independent_keys,
            test_labels=labels,
            batch_size=args.batch_size,
            threshold=args.threshold,
            save_histograms=bool(args.save_histograms),
            ensemble_test=args.ensemble_test,
            max_workers=args.max_workers,
            num_data_workers=args.num_data_workers,
            output_dir=args.output_dir,
            pkl_dir=dataset_pkls_dir,
            return_logits=True,
            cdf_bins=args.cdf_bins,
            test_type=TestType.BOTH,
            logger=mlflow,
            seed=args.seed,
            draw_pvalues_trend_figure=bool(args.draw_pvalues_trend_figure),
            reference_cache_suffix=reference_cache_suffix,
            cache_suffix=inference_cache_suffix,

        )

        balance_labels, balance_scores = balanced_testset(results['labels'], results['scores'], random_state=42)
        auc = plot_roc_curve(balance_labels, balance_scores, test_id, args.output_dir)
        ap = average_precision_score(balance_labels, balance_scores)
        plot_fakeness_score_histogram(results, test_id, args.output_dir, args.threshold)
        export_score_bins_table(results['scores'], results['labels'], test_id, args.output_dir, logger=mlflow)
        mlflow.log_metric("AUC", auc)
        mlflow.log_metric("AP", ap)


if __name__ == "__main__":
    main()
