import torch
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold, train_test_split

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers


from embeddings_model.settings import LOGS_DIR
from embeddings_model.learning.regressor import Regressor
from embeddings_model.learning.classifier import Classifier
from embeddings_model.utils.callbacks.time import TimingCallback

def train_test(datamodule, model, epochs=6, lr=1e-2, experiment_name='default', regression=False,
            use_cuda=False, test_fold=None, return_dict=False, test_repetitions=1):
    """ Train model and return predictions for test dataset"""
    train_loader = datamodule.train_dataloader(test_fold=test_fold)
    val_loader = datamodule.val_dataloader(test_fold=test_fold)
    test_loader = datamodule.test_dataloader(test_fold=test_fold)

    if regression:
        model = Regressor(model=model, lr=lr)
    else:
        class_dims = datamodule.class_dims
        model = Classifier(model=model, lr=lr, class_dims=class_dims)

    csv_logger = pl_loggers.CSVLogger(LOGS_DIR, name=experiment_name)
    checkpoint_callback = ModelCheckpoint(dirpath='./checkpoints/',
        save_top_k=1,
        monitor='valid_loss',
        mode='min'
    )
    
    _use_cuda = use_cuda and torch.cuda.is_available()
    timing_callback = TimingCallback()
    
    training_start_time = time.time()
    trainer = pl.Trainer(gpus=1 if _use_cuda else 0, max_epochs=epochs, progress_bar_refresh_rate=20,
                        #profiler="simple", 
                        logger=[csv_logger],
                        callbacks=[checkpoint_callback, timing_callback])
    trainer.fit(model, train_loader, val_loader)
    
    training_time = time.time() - training_start_time

    checkpoint = torch.load(checkpoint_callback.best_model_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    testing_times = []
    for _ in range(test_repetitions):
        test_start_time = time.time()
        
        test_predictions = [] 
        true_labels = []
        with torch.no_grad():
            for batch_text_X, batch_text_y in test_loader:
                batch_text_X, batch_text_y = batch_text_X, batch_text_y

                test_predictions.append(model(batch_text_X))
                true_labels.append(batch_text_y)

        testing_time = time.time() - test_start_time

        testing_times.append(testing_time)

    mean_testing_time = np.mean(testing_times)


    test_predictions = torch.cat(test_predictions, dim=0).cpu().numpy()
    true_labels = torch.cat(true_labels, dim=0).cpu().numpy()
    if not regression:
        true_labels = true_labels.astype(int)
    
    times = {
        'training_time': training_time, 
        'testing_time': mean_testing_time,
        'testing_times': testing_times
        }

    if return_dict:
        return {
            'test_predictions': test_predictions, 
            'true_labels': true_labels, 
            'model': model, 
            'times': times,
            'timing_callback': timing_callback
            }
    else:
        return test_predictions, true_labels, model, times
