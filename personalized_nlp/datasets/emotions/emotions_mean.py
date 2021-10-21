import os
import pickle
from typing import Optional, List

import pandas as pd
import torch
from torch.utils.data import DataLoader, dataset
import pytorch_lightning as pl

from personalized_nlp.settings import FASTTEXT_EMBEDDINGS, STORAGE_DIR, TRANSFORMERS_EMBEDDINGS
from personalized_nlp.datasets.dataset import BatchIndexedDataset
from personalized_nlp.utils.tokenizer import get_text_data
from personalized_nlp.utils.biases import get_annotator_biases
from personalized_nlp.utils.data_splitting import split_texts
from personalized_nlp.datasets.datamodule_base import BaseDataModule
from personalized_nlp.utils.embeddings import create_embeddings



import torch.utils.data

class MeanDataset(torch.utils.data.Dataset):

    def __init__(self, X, y) -> None:
        super().__init__()
        self.X = list(X[i] for i in range(len(X)))
        self.y = y.values.reshape(-1, 10)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index):
        x = torch.tensor(self.X[index]).unsqueeze(0)
        return {'embeddings': x}, self.y[index]



class EmotionsMeanDataModule(pl.LightningDataModule):

    def __init__(
                self,
                data_dir: str = STORAGE_DIR / 'emotions_data/texts_mean/texts',
                batch_size: int = 3000,
                embeddings_type: str = 'bert',
                language: str = 'english',
                normalize=False,
                check_path: bool = True,
                **kwargs,
        ):
            super().__init__()

            self.folds_num = 10
            self.data_dir = data_dir
            self.data = {
                'train': pd.read_csv(self.data_dir / 'cawi1_mean_train.csv'),
                'test': pd.read_csv(self.data_dir / 'cawi1_mean_test.csv'),
                'val': pd.read_csv(self.data_dir / 'cawi1_mean_test.csv')
            }

            self.data_url = None
            self.batch_size = batch_size
            self.language = language
            self.embeddings_type = embeddings_type
            self.annotation_column = ['OCZEKIWANIE',
                                    'POBUDZENIE',
                                    'RADOŚĆ',
                                    'SMUTEK',
                                    'STRACH',
                                    'WSTRĘT',
                                    'ZASKOCZENIE',
                                    'ZAUFANIE',
                                    'ZNAK EMOCJI',
                                    'ZŁOŚĆ']
            self.text_column = 'text'

            self.embeddings_path = STORAGE_DIR / \
                f'emotions_data/texts_mean/embeddings/'

            self.train_split_names = ['present', 'past']
            self.val_split_names = ['future1']
            self.test_split_names = ['future2']

            self.normalize = normalize
            self._create_embeddings()

    @property
    def words_number(self):
        return self.tokens_sorted.max() + 1

    @property
    def class_dims(self):
        return [5] * 8 + [7, 5]

    @property
    def text_embedding_dim(self):
        if self.embeddings_type in ['xlmr', 'bert']:
            return 768
        elif self.embeddings_type in FASTTEXT_EMBEDDINGS:
            return 300
        else:
            return 1024

    def _create_embeddings(self):
        embeddings_path = self.embeddings_path

        if self.embeddings_type == 'xlmr':
            model_name = 'xlm-roberta-base'
        elif self.embeddings_type == 'bert':
            model_name = 'bert-base-cased'
        elif self.embeddings_type == 't5':
            model_name = 'google/t5-large-ssm'
        elif self.embeddings_type == 'deberta':
            model_name = 'microsoft/deberta-large'
        elif self.embeddings_type == 'labse':
            model_name = 'sentence-transformers/LaBSE'
        elif self.embeddings_type == 'glove':
            model_name = 'glove'
        elif self.embeddings_type == 'skipgram':
            model_name = 'skipgram'
        elif self.embeddings_type == 'cbow':
            model_name = 'cbow'
        else:
            raise NotImplementedError(f'{self.embeddings_type} is not implemented')
        
        is_transformer = self.embeddings_type in TRANSFORMERS_EMBEDDINGS

        use_cuda = torch.cuda.is_available()
        for split_name, split in self.data.items():
            create_embeddings(
                list(split[self.text_column]), 
                embeddings_path / f'cawi1_mean_{split_name}_{self.embeddings_type}.p',
                model_name=model_name, 
                is_transformer=is_transformer, 
                use_cuda=use_cuda, 
                model=None,
                pickle_embeddings=True)

    def _prepare_dataset(self, split: str):
        with open(self.embeddings_path / f'cawi1_mean_{split}_{self.embeddings_type}.p', 'rb') as f:
            X = pickle.load(f)
        #print(X[0])
        #raise None
        y = self.data[split][self.annotation_column]
        return MeanDataset(X, y)
        

    def _prepare_dataloader(self, dataset, shuffle=True):
        if shuffle:
            sampler = torch.utils.data.sampler.BatchSampler(
                torch.utils.data.sampler.RandomSampler(dataset),
                batch_size=self.batch_size,
                drop_last=False)
        else:
            sampler = torch.utils.data.sampler.BatchSampler(
                torch.utils.data.sampler.SequentialSampler(dataset),
                batch_size=self.batch_size,
                drop_last=False)

        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)

    def train_dataloader(self, test_fold=None) -> DataLoader:
        dataset = self._prepare_dataset('train')
        return self._prepare_dataloader(dataset)

    def val_dataloader(self, test_fold=None) -> DataLoader:
        dataset = self._prepare_dataset('val')
        return self._prepare_dataloader(dataset, shuffle=False)

    def test_dataloader(self, test_fold=None) -> DataLoader:
        dataset = self._prepare_dataset('test')
        return self._prepare_dataloader(dataset, shuffle=False)
