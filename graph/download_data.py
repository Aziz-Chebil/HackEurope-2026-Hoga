"""Download TwiBot-20 dataset from Kaggle."""

import kagglehub


def download_twibot20() -> str:
    """Download TwiBot-20 dataset and return the path to the data files.

    Requires KAGGLE_USERNAME and KAGGLE_KEY environment variables,
    or a ~/.kaggle/kaggle.json file.
    """
    print("Downloading TwiBot-20 dataset from Kaggle...")
    path = kagglehub.dataset_download("marvinvanbo/twibot-20")
    print(f"Dataset downloaded to: {path}")
    return path


if __name__ == "__main__":
    download_twibot20()
