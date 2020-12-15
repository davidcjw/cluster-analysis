from dataclasses import dataclass, field
from typing import List, Union
import pandas as pd
import numpy as np


@dataclass
class Cluster:

    df: pd.DataFrame = field(
        metadata={"help": "DataFrame or numpy array of X values"}
    )
    labels: Union[pd.Series, List[int], np.ndarray] = field(
        metadata={"help": "Cluster labels. Must have at least 2 classes"}
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


if __name__ == "__main__":
    from sklearn.cluster import KMeans
    df = pd.read_csv("data.csv").select_dtypes("int")
    km = KMeans()
    km.fit(df)
    cluster = Cluster(df, km.labels_)
    assert cluster.df is not None and isinstance(cluster.df, pd.DataFrame)
    assert cluster.labels is not None and isinstance(cluster.labels, pd.Series)
