
from embeddings_model.models.baseline import Net
from embeddings_model.models.onehot import NetOneHot
from embeddings_model.models.human_bias import HumanBiasNet
from embeddings_model.models.bias import AnnotatorBiasNet
from embeddings_model.models.annotator import AnnotatorEmbeddingNet
from embeddings_model.models.annotator_word import AnnotatorWordEmbeddingNet

models = {
    'baseline': Net,
    'onehot': NetOneHot, 
    'peb': HumanBiasNet, 
    'bias': AnnotatorBiasNet, 
    'embedding': AnnotatorEmbeddingNet, 
    'word_embedding': AnnotatorWordEmbeddingNet
}