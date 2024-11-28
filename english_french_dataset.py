import pandas as pd
from torch.utils.data import Dataset


class EnglishFrenchDataset(Dataset):
    def __init__(
        self,
        filename: str,
        english_to_french: bool = True,
    ) -> None:
        self.filename = filename
        self.english_to_french = english_to_french

    def __len__(self) -> int:
        with open(self.filename, "r") as f:
            for i, line in enumerate(f):
                pass
        return i

    def __getitem__(self, idx: int) -> tuple[str, str]:
        df = pd.read_csv(
            self.filename,
            header=0,
            names=["en", "fr"],
            nrows=1,
            skiprows=idx,
        ).iloc[0]
        text_source = df["en"] if self.english_to_french else df["fr"]
        text_destination = df["fr"] if self.english_to_french else df["en"]
        return (text_source, text_destination)
