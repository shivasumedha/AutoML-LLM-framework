import pandas as pd


def load_dataset(file_path):
    """
    Load CSV dataset.
    """

    try:
        df = pd.read_csv(file_path)

        print("Dataset loaded successfully!")
        print(f"Dataset shape: {df.shape}")

        return df

    except FileNotFoundError:
        print(f"ERROR: File not found: {file_path}")
        return None

    except Exception as e:
        print(f"ERROR while loading dataset: {e}")
        return None