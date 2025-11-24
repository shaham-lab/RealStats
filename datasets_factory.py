from enum import Enum
from data_utils import (
    CocoDataset,
    CrossDomainRealDataset,
    EmptyDataset,
    ImageDataset,
    RealStatsDataset,
    ProGanDataset,
)


class RealStatsGenerators(Enum):
    CNNSPOT_BIGGAN = "CNNSpot_test/biggan"
    CNNSPOT_CRN = "CNNSpot_test/crn"
    CNNSPOT_CYCLEGAN = "CNNSpot_test/cyclegan"
    CNNSPOT_DEEPFAKE = "CNNSpot_test/deepfake"
    CNNSPOT_GAUGAN = "CNNSpot_test/gaugan"
    CNNSPOT_IMLE = "CNNSpot_test/imle"
    CNNSPOT_PROGAN = "CNNSpot/progan"
    CNNSPOT_SAN = "CNNSpot_test/san"
    CNNSPOT_STARGAN = "CNNSpot_test/stargan"
    CNNSPOT_STYLEGAN2 = "CNNSpot_test/stylegan2"
    CNNSPOT_WHICHFACEISREAL = "CNNSpot_test/whichfaceisreal"
    GENIMAGE_ADM = "GenImage/adm_genimage"
    GENIMAGE_MIDJOURNEY = "GenImage/midjourney_genimage"
    GENIMAGE_SD_V4 = "GenImage/sd_v4_genimage"
    GENIMAGE_SD_V5 = "GenImage/sd_v5_genimage"
    GENIMAGE_VQDM = "GenImage/vqdm_genimage"
    GENIMAGE_WUKONG = "GenImage/wukong_genimage"
    STABLE_DIFFUSION_FACES_SD2 = "stable_diffusion_human_models/768_sdv2"
    STABLE_DIFFUSION_FACES_SDXL = "stable_diffusion_human_models/1024_sdxl"
    SYNTHBUSTER_DALLE3 = "SynthBuster/dalle3"
    SYNTHBUSTER_MIDJOURNEY_V5 = "SynthBuster/midjourney-v5"
    SYNTHBUSTER_STABLE_DIFFUSION_2 = "SynthBuster/stable-diffusion-2"
    SYNTHBUSTER_STABLE_DIFFUSION_XL = "SynthBuster/stable-diffusion-xl"
    UNIVERSAL_FAKE_DETECT_DALLE = "Universal_Fake_Detect/diffusion_datasets/dalle"
    UNIVERSAL_FAKE_DETECT_GLIDE_100_27 = "Universal_Fake_Detect/diffusion_datasets/glide_100_27"
    UNIVERSAL_FAKE_DETECT_GLIDE_50_27 = "Universal_Fake_Detect/diffusion_datasets/glide_50_27"
    CELEBA = "CelebA"


def _dataset_entry(generator=None):
    return {
        "reference_real": {
            "path": "data/RealStatsDataset",
            "class": lambda root, label, transform=None: RealStatsDataset(
                root, "reference_real_paths.csv", label, transform
            ),
        },
        "test_real": {
            "path": "data/RealStatsDataset",
            "class": lambda root, label, transform=None: RealStatsDataset(
                root, "test_real_paths.csv", label, transform
                # root, "celeba_real_paths.csv", label, transform
            ),
        },
        "test_fake": {
            "path": "data/RealStatsDataset",
            "class": lambda root, label, transform=None: RealStatsDataset(
                root, "test_fake_paths.csv", label, transform, generator=generator
            ),
        },
    }


def _cross_domain_real_only_entry(test_csv):
    return {
        "reference_real": {
            "path": "data/RealStatsDataset",
            "class": lambda root, label, transform=None: RealStatsDataset(
                # root, "reference_real_paths.csv", label, transform
                # root, "all_real_paths.csv", label, transform
                # root, "real_paths_distributional_shift_per1.csv", label, transform
                # root, "real_paths_distributional_shift_per5.csv", label, transform
                # root, "real_paths_distributional_shift_per10.csv", label, transform
                # root, "real_paths_distributional_shift_per20.csv", label, transform
                # root, "real_paths_distributional_shift_per30.csv", label, transform
                root, "reference_real_paths_no_faces_or_persons.csv", label, transform
            ),
        },
        "test_real": {
            "path": "data/RealStatsDataset",
            "class": lambda root, label, transform=None: CrossDomainRealDataset(
                root, test_csv, label, transform
            ),
        },
        "test_fake": {
            "path": "data/RealStatsDataset",
            "class": lambda root, label, transform=None: EmptyDataset(
                label=label, transform=transform
            ),
        },
    }

class DatasetType(Enum):
    # === Base dataset ===
    ALL = _dataset_entry()

    # === Cross-domain real-only experiment ===
    CELEBA_REAL_ONLY = _cross_domain_real_only_entry("celeba_real_paths.csv")

    # ===  CNNSpot_test ===
    CNNSPOT_BIGGAN = _dataset_entry(RealStatsGenerators.CNNSPOT_BIGGAN.value)
    CNNSPOT_CRN = _dataset_entry(RealStatsGenerators.CNNSPOT_CRN.value)
    CNNSPOT_CYCLEGAN = _dataset_entry(RealStatsGenerators.CNNSPOT_CYCLEGAN.value)
    CNNSPOT_DEEPFAKE = _dataset_entry(RealStatsGenerators.CNNSPOT_DEEPFAKE.value)
    CNNSPOT_GAUGAN = _dataset_entry(RealStatsGenerators.CNNSPOT_GAUGAN.value)
    CNNSPOT_IMLE = _dataset_entry(RealStatsGenerators.CNNSPOT_IMLE.value)
    CNNSPOT_PROGAN = _dataset_entry(RealStatsGenerators.CNNSPOT_PROGAN.value) 
    CNNSPOT_SAN = _dataset_entry(RealStatsGenerators.CNNSPOT_SAN.value)
    CNNSPOT_STARGAN = _dataset_entry(RealStatsGenerators.CNNSPOT_STARGAN.value) 
    CNNSPOT_STYLEGAN2 = _dataset_entry(RealStatsGenerators.CNNSPOT_STYLEGAN2.value)
    CNNSPOT_WHICHFACEISREAL = _dataset_entry(RealStatsGenerators.CNNSPOT_WHICHFACEISREAL.value) 

    # === GenImage ===
    GENIMAGE_ADM = _dataset_entry(RealStatsGenerators.GENIMAGE_ADM.value)
    GENIMAGE_MIDJOURNEY = _dataset_entry(RealStatsGenerators.GENIMAGE_MIDJOURNEY.value)
    GENIMAGE_SD_V4 = _dataset_entry(RealStatsGenerators.GENIMAGE_SD_V4.value)
    GENIMAGE_SD_V5 = _dataset_entry(RealStatsGenerators.GENIMAGE_SD_V5.value)
    GENIMAGE_VQDM = _dataset_entry(RealStatsGenerators.GENIMAGE_VQDM.value)
    GENIMAGE_WUKONG = _dataset_entry(RealStatsGenerators.GENIMAGE_WUKONG.value)

    # === Stable Diffusion Faces Models ===
    STABLE_DIFFUSION_FACES_SD2 = _dataset_entry(RealStatsGenerators.STABLE_DIFFUSION_FACES_SD2.value)
    STABLE_DIFFUSION_FACES_SDXL = _dataset_entry(RealStatsGenerators.STABLE_DIFFUSION_FACES_SDXL.value)

    # === SynthBuster ===
    SYNTHBUSTER_DALLE3 = _dataset_entry(RealStatsGenerators.SYNTHBUSTER_DALLE3.value)
    SYNTHBUSTER_MIDJOURNEY_V5 = _dataset_entry(RealStatsGenerators.SYNTHBUSTER_MIDJOURNEY_V5.value)
    SYNTHBUSTER_STABLE_DIFFUSION_2 = _dataset_entry(RealStatsGenerators.SYNTHBUSTER_STABLE_DIFFUSION_2.value)
    SYNTHBUSTER_STABLE_DIFFUSION_XL = _dataset_entry(RealStatsGenerators.SYNTHBUSTER_STABLE_DIFFUSION_XL.value)

    # === Universal Fake Detect ===
    UNIVERSAL_FAKE_DETECT_DALLE = _dataset_entry(RealStatsGenerators.UNIVERSAL_FAKE_DETECT_DALLE.value)
    UNIVERSAL_FAKE_DETECT_GLIDE_100_27 = _dataset_entry(RealStatsGenerators.UNIVERSAL_FAKE_DETECT_GLIDE_100_27.value)
    UNIVERSAL_FAKE_DETECT_GLIDE_50_27 = _dataset_entry(RealStatsGenerators.UNIVERSAL_FAKE_DETECT_GLIDE_50_27.value)

    def get_paths(self):
        return self.value


# Dataset Factory
class DatasetFactory:
    """Factory class for creating datasets based on dataset type."""

    @staticmethod
    def create_dataset(dataset_type, transform=None):
        """
        Create datasets dynamically based on dataset type.

        Args:
            dataset_type (str): The dataset type (e.g., 'ALL').
            transform (callable, optional): Transform to apply to the images.

        Returns:
            dict: Dictionary containing dataset instances for reference_real, test_real, test_fake
        """
        dataset_type = dataset_type.upper()

        if dataset_type not in DatasetType.__members__:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

        dataset_info = DatasetType[dataset_type].get_paths()

        datasets = {}
        for split, split_info in dataset_info.items():
            dataset_class = split_info["class"]
            dataset_path = split_info["path"]

            datasets[split] = dataset_class(dataset_path, 0 if "real" in split else 1, transform=transform)

        return datasets
