from typing import Sequence

from personalized_nlp.settings import STORAGE_DIR
from personalized_nlp.datasets.datamodule_base import BaseDataModule


class CaviDataset(BaseDataModule):

    def __init__(
            self,
            data_dir: str = STORAGE_DIR / 'cavi/texts',
            batch_size: int = 3000,
            embeddings_type: str = 'bert',
            language: str = 'english',
            split_sizes: Sequence[float or int] = (0.55, 0.15, 0.15, 0.15),
            normalize=False,
            **kwargs
    ):
        super(CaviDataset, self).__init__(**kwargs)
        pass
