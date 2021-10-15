from torch.random import seed
from personalized_nlp.datasets.emotions.emotions import EmotionsDataModule
from personalized_nlp.datasets.wiki.aggression import AggressionDataModule
from personalized_nlp.datasets.wiki.attack import AttackDataModule
from personalized_nlp.datasets.wiki.toxicity import ToxicityDataModule
from pytorch_lightning import loggers as pl_loggers
from personalized_nlp.experiments.persemo_experiments import EMBEDDING_TYPES
from personalized_nlp.settings import STORAGE_DIR, LOGS_DIR, TRANSFORMERS_EMBEDDINGS, FASTTEXT_EMBEDDINGS
from personalized_nlp.learning.train import train_test
from personalized_nlp.models.annotator import AnnotatorEmbeddingNet
from personalized_nlp.models.baseline import Net
import random
import numpy as np
import torch
import os
import argparse


BASE_MODULE = EmotionsDataModule
TARGET_MODULES = [ToxicityDataModule, AggressionDataModule, AttackDataModule]
PROJECT_NAME = "Cawi2AgressionTransferLearning"

EMBEDDING_TYPES = FASTTEXT_EMBEDDINGS + TRANSFORMERS_EMBEDDINGS

def seed_everything():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)


if __name__ == '__main__':
    seed_everything()
    results = []
    for TARGET_MODULE in TARGET_MODULES:
        for embeddings_type in EMBEDDING_TYPES:
            base_module = BASE_MODULE(embeddings_type=embeddings_type, normalize=True,
                                            batch_size=1000, language='english')
            base_module.prepare_data()
            base_module.setup()
            base_module.compute_word_stats(
                min_word_count=200,
                min_std=0.0,
                words_per_text=100
            )

            for embedding_dim in [50]:
                hparams = {
                    'dataset': type(base_module).__name__,
                    'model_type': 'baseline',
                    'embeddings_type': embeddings_type,
                    'embedding_size': embedding_dim,
                    'fold_num': 1,
                    'regression': True,
                }
                logger = pl_loggers.WandbLogger(save_dir=LOGS_DIR, config=hparams, project=PROJECT_NAME, 
                    log_model=False)

                output_dim = len(base_module.class_dims)
                text_embedding_dim = base_module.text_embedding_dim
                base_model = Net(
                    output_dim=output_dim,
                    text_embedding_dim=text_embedding_dim
                )
                model = train_test(base_module, base_model, epochs=20, lr=0.008,
                            experiment_name='default', regression=True,
                            use_cuda=True, test_fold=1, logger=logger)

                target_module = TARGET_MODULE(embeddings_type=embeddings_type, normalize=True,
                                            batch_size=1000, model=model)
                target_module.prepare_data()
                target_module.setup()
                target_module.compute_word_stats(
                    min_word_count=200,
                    min_std=0.0,
                    words_per_text=100
                )
                logger.experiment.finish()
                for fold_num in range(5):
                    hparams = {
                        'dataset': type(target_module).__name__,
                        'model_type': 'embedding',
                        'embeddings_type': embeddings_type,
                        'embedding_size': embedding_dim,
                        'fold_num': fold_num,
                        'regression': False,
                    }
                    logger = pl_loggers.WandbLogger(
                                save_dir=LOGS_DIR, config=hparams, project=PROJECT_NAME, log_model=False)

                    target_output_dim = 2
                    text_embedding_dim = target_module.text_embedding_dim
                    target_model = AnnotatorEmbeddingNet(output_dim=target_output_dim, text_embedding_dim=text_embedding_dim,
                                                    word_num=target_module.words_number,
                                                    annotator_num=target_module.annotators_number, dp=0.0, dp_emb=0.25,
                                                    embedding_dim=embedding_dim, hidden_dim=100)
                    train_test(target_module, target_model, epochs=20, lr=0.008,
                                experiment_name='default', regression=False,
                                use_cuda=True, test_fold=fold_num, logger=logger)

                    logger.experiment.finish()

        

