import random
import numpy as np
import torch

from personalized_nlp.datasets.emotions.emotions import EmotionsDataModule
from personalized_nlp.datasets.wiki.aggression import AggressionDataModule
from personalized_nlp.datasets.wiki.attack import AttackDataModule
from personalized_nlp.datasets.wiki.toxicity import ToxicityDataModule

from personalized_nlp.settings import LOGS_DIR, TRANSFORMERS_EMBEDDINGS
from personalized_nlp.transfer_learning_methods import train_test_base_model, train_test_random_transfer, train_test_transfer_learning


def seed_everything():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

EMBEDDINGS_TYPES = TRANSFORMERS_EMBEDDINGS
TARGET_MODULES = [AttackDataModule, AggressionDataModule, ToxicityDataModule]
PROJECT_NAME = 'Emotions->Wikipedia Transfer Learning'        

if __name__ == '__main__':
    # train_test_base_model(
    #     embeddings=EMBEDDINGS_TYPES,
    #     datamodules=TARGET_MODULES,
    #     logs_dir=LOGS_DIR,
    #     experiment_name='base_method',
    #     epochs=5,
    #     lr=0.008,
    #     project_name=PROJECT_NAME
    # )
    # train_test_random_transfer(
    #     embeddings=EMBEDDINGS_TYPES,
    #     datamodules=TARGET_MODULES,
    #     logs_dir=LOGS_DIR,
    #     experiment_name='random_feature_generator',
    #     num_features=[10, 20],
    #     epochs=5,
    #     lr=0.008,
    #     project_name=PROJECT_NAME
    # )
    train_test_transfer_learning(
        embeddings=EMBEDDINGS_TYPES,
        datamodules=TARGET_MODULES,
        logs_dir=LOGS_DIR,
        experiment_name='transfer_learning',
        epochs=5,
        lr=0.008,
        project_name=PROJECT_NAME
    )
