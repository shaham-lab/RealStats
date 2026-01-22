import argparse
import json
import os
import random

from datasets_factory import DatasetFactory, DatasetType


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Split DeepFakeBenchmark real samples into reference/test and save JSON splits."
        )
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default=DatasetType.DEEPFAKE_BENCHMARK.name,
        choices=[e.name for e in DatasetType],
        help="Dataset type to load (default: DEEPFAKE_BENCHMARK).",
    )
    parser.add_argument(
        "--reference_ratio",
        type=float,
        default=0.3,
        help="Fraction of real samples to allocate to reference split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for shuffling real samples.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="dfdc_real_splits.json",
        help="Path to write JSON split file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    datasets = DatasetFactory.create_dataset(dataset_type=args.dataset_type, transform=None)
    reference_dataset = datasets["reference_real"]
    test_real_dataset = datasets["test_real"]
    test_fake_dataset = datasets["test_fake"]

    real_paths = list(getattr(reference_dataset, "image_paths", [])) + list(
        getattr(test_real_dataset, "image_paths", [])
    )
    fake_paths = list(getattr(test_fake_dataset, "image_paths", []))

    rng = random.Random(args.seed)
    rng.shuffle(real_paths)

    reference_count = int(len(real_paths) * args.reference_ratio)
    reference_paths = real_paths[:reference_count]
    test_real_paths = real_paths[reference_count:]

    output_data = {
        "reference_real": reference_paths,
        "test_real": test_real_paths,
        "test_fake": fake_paths,
    }

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(output_data, f, indent=2)

    print(
        "Wrote JSON splits to {path}: reference={ref}, test_real={real}, test_fake={fake}".format(
            path=args.output_json,
            ref=len(reference_paths),
            real=len(test_real_paths),
            fake=len(fake_paths),
        )
    )


if __name__ == "__main__":
    main()
