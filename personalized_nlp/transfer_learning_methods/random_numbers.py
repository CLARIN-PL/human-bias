from typing import Any, List

import pytorch_lightning as pl
import torch
from torch import nn
from pytorch_lightning.loggers import WandbLogger
from personalized_nlp.learning.train import train_test
from personalized_nlp.models.annotator_word import AnnotatorWordEmbeddingNet

class RandomModel(nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features

    def forward(self, x: Any) -> torch.Tensor:
        return torch.rand((1, self.num_features)).cuda()

def train_test_random_transfer(    
    embeddings: List[str],
    datamodules: List[pl.LightningDataModule.__class__],
    logs_dir: str,
    experiment_name: str,
    epochs: int,
    lr: float,
    num_features: List[int],
    project_name: str):

    regression=False
    for datamodule in datamodules:
        for features in num_features:
            random_model = RandomModel(features)
            for embeddings_type in embeddings:
                module = datamodule(
                    embeddings_type=embeddings_type,
                    prev_model=random_model,
                    num_features=features,
                    check_path=False
                ) 
                module.prepare_data()
                module.setup()
                for embedding_dim in [50]:
                    for fold_num in range(5):
                        hparams = {
                                'dataset': type(module).__name__,
                                'model_type': 'random_generator',
                                'embeddings_type': embeddings_type,
                                'embedding_size': embedding_dim,
                                'num_features': features,
                                'fold_num': fold_num,
                                'regression': regression,
                            }
                        logger = WandbLogger(
                            save_dir=logs_dir, config=hparams, project=project_name, log_model=False)

                        output_dim = 2
                        text_embedding_dim = module.text_embedding_dim
                        model = AnnotatorWordEmbeddingNet(output_dim=output_dim, text_embedding_dim=text_embedding_dim,
                                        word_num=module.words_number,
                                        annotator_num=module.annotators_number, dp=0.0, dp_emb=0.25,
                                        embedding_dim=embedding_dim, hidden_dim=100)
                        train_test(module, model, epochs=epochs, lr=lr,
                                    experiment_name=experiment_name, regression=regression,
                                    use_cuda=True, logger=logger)

                        logger.experiment.finish()

