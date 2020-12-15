from dataclasses import dataclass, field
from typing import List
import pandas as pd


@dataclass
class Cluster:

    labels: List[int] = field(
        metadata={"help": "Cluster labels. Must have at least 2 classes"}
    )
    df: pd.DataFrame = field(
        metadata={"help": "DataFrame or numpy array of X values"}
    )

    def __post_init__(self):
        self._check_input()

    def stats(self):
        pass

    def _check_input(self):
        if not isinstance(self.labels, pd.Series):
            self.labels = pd.Series(self.labels)

        if not isinstance(self.df, pd.DataFrame):
            raise ValueError("df needs to be a DataFrame or numpy array!")

        if self.labels.size != self.df.shape[0]:
            raise ValueError("df and labels have to be of same length!")

        if self.labels.nunique() < 2:
            raise ValueError("Labels need more than 1 cluster!")

    @property
    def df(self):
        return self.df

    @df.setter
    def df(self, col, val):
        print(f"Setting column {col} to value {val}...")
        setattr(self, self.df, val)


if __name__ == "__main__":
    from sklearn.cluster import KMeans
    df = pd.read_csv("data.csv").select_dtypes("int")
    km = KMeans().fit("data.csv")
    cluster = Cluster(df, km.labels_)
