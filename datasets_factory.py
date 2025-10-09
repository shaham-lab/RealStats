from enum import Enum
from data_utils import CocoDataset, CSVDataset, ImageDataset, ProGanDataset


class CsvDatasetGenerator(Enum):
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


def _csv_dataset_entry(generator=None):
    return {
        "reference_real": {
            "path": "data/csv_datasets",
            "class": lambda root, label, transform=None: CSVDataset(
                root, "reference_real_paths.csv", label, transform
            ),
        },
        "test_real": {
            "path": "data/csv_datasets",
            "class": lambda root, label, transform=None: CSVDataset(
                root, "test_real_paths.csv", label, transform
            ),
        },
        "test_fake": {
            "path": "data/csv_datasets",
            "class": lambda root, label, transform=None: CSVDataset(
                root, "test_fake_paths.csv", label, transform, generator=generator
            ),
        },
    }

class DatasetType(Enum):
    # === Base dataset ===
    ALL = _csv_dataset_entry()

    # ===  CNNSpot_test ===
    CNNSPOT_BIGGAN = _csv_dataset_entry(CsvDatasetGenerator.CNNSPOT_BIGGAN.value)
    CNNSPOT_CRN = _csv_dataset_entry(CsvDatasetGenerator.CNNSPOT_CRN.value)
    CNNSPOT_CYCLEGAN = _csv_dataset_entry(CsvDatasetGenerator.CNNSPOT_CYCLEGAN.value)
    CNNSPOT_DEEPFAKE = _csv_dataset_entry(CsvDatasetGenerator.CNNSPOT_DEEPFAKE.value)
    CNNSPOT_GAUGAN = _csv_dataset_entry(CsvDatasetGenerator.CNNSPOT_GAUGAN.value)
    CNNSPOT_IMLE = _csv_dataset_entry(CsvDatasetGenerator.CNNSPOT_IMLE.value)
    CNNSPOT_PROGAN = _csv_dataset_entry(CsvDatasetGenerator.CNNSPOT_PROGAN.value)
    CNNSPOT_SAN = _csv_dataset_entry(CsvDatasetGenerator.CNNSPOT_SAN.value)
    CNNSPOT_STARGAN = _csv_dataset_entry(CsvDatasetGenerator.CNNSPOT_STARGAN.value)
    CNNSPOT_STYLEGAN2 = _csv_dataset_entry(CsvDatasetGenerator.CNNSPOT_STYLEGAN2.value)
    CNNSPOT_WHICHFACEISREAL = _csv_dataset_entry(CsvDatasetGenerator.CNNSPOT_WHICHFACEISREAL.value)

    # === GenImage ===
    GENIMAGE_ADM = _csv_dataset_entry(CsvDatasetGenerator.GENIMAGE_ADM.value)
    GENIMAGE_MIDJOURNEY = _csv_dataset_entry(CsvDatasetGenerator.GENIMAGE_MIDJOURNEY.value)
    GENIMAGE_SD_V4 = _csv_dataset_entry(CsvDatasetGenerator.GENIMAGE_SD_V4.value)
    GENIMAGE_SD_V5 = _csv_dataset_entry(CsvDatasetGenerator.GENIMAGE_SD_V5.value)
    GENIMAGE_VQDM = _csv_dataset_entry(CsvDatasetGenerator.GENIMAGE_VQDM.value)
    GENIMAGE_WUKONG = _csv_dataset_entry(CsvDatasetGenerator.GENIMAGE_WUKONG.value)

    # === Stable Diffusion Faces Models ===
    STABLE_DIFFUSION_FACES_SD2 = _csv_dataset_entry(CsvDatasetGenerator.STABLE_DIFFUSION_FACES_SD2.value)
    STABLE_DIFFUSION_FACES_SDXL = _csv_dataset_entry(CsvDatasetGenerator.STABLE_DIFFUSION_FACES_SDXL.value)

    # === SynthBuster ===
    SYNTHBUSTER_DALLE3 = _csv_dataset_entry(CsvDatasetGenerator.SYNTHBUSTER_DALLE3.value)
    SYNTHBUSTER_MIDJOURNEY_V5 = _csv_dataset_entry(CsvDatasetGenerator.SYNTHBUSTER_MIDJOURNEY_V5.value)
    SYNTHBUSTER_STABLE_DIFFUSION_2 = _csv_dataset_entry(CsvDatasetGenerator.SYNTHBUSTER_STABLE_DIFFUSION_2.value)
    SYNTHBUSTER_STABLE_DIFFUSION_XL = _csv_dataset_entry(CsvDatasetGenerator.SYNTHBUSTER_STABLE_DIFFUSION_XL.value)

    # === Universal Fake Detect ===
    UNIVERSAL_FAKE_DETECT_DALLE = _csv_dataset_entry(CsvDatasetGenerator.UNIVERSAL_FAKE_DETECT_DALLE.value)
    UNIVERSAL_FAKE_DETECT_GLIDE_100_27 = _csv_dataset_entry(CsvDatasetGenerator.UNIVERSAL_FAKE_DETECT_GLIDE_100_27.value)
    UNIVERSAL_FAKE_DETECT_GLIDE_50_27 = _csv_dataset_entry(CsvDatasetGenerator.UNIVERSAL_FAKE_DETECT_GLIDE_50_27.value)

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

        # Ensure dataset_type exists in DatasetType
        if dataset_type not in DatasetType.__members__:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

        # Retrieve dataset configuration
        dataset_info = DatasetType[dataset_type].get_paths()

        datasets = {}
        for split, split_info in dataset_info.items():
            dataset_class = split_info["class"]
            dataset_path = split_info["path"]

            datasets[split] = dataset_class(dataset_path, 0 if "real" in split else 1, transform=transform)

        return datasets
