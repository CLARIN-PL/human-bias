from types import new_class
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from transformers import AutoTokenizer, AutoModelForPreTraining
import torch
from personalized_nlp.settings import TRANSFORMERS_EMBEDDINGS, FASTTEXT_EMBEDDINGS, FASTTEXT_MODELS_PATHS
import fasttext

from tqdm import tqdm
import pickle

def _batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def _get_embeddings_transformers(texts, model_name: str, max_seq_len=256, use_cuda=False) -> torch.Tensor:
    if use_cuda:
        device = 'cuda'
    else:
        device = 'cpu'

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    all_embeddings = []
    for batched_texts in tqdm(_batch(texts, 200), total=len(texts)/200):
        with torch.no_grad():
            batch_encoding = tokenizer.batch_encode_plus(
                batched_texts,
                padding='longest',
                add_special_tokens=True,
                truncation=True, max_length=max_seq_len,
                return_tensors='pt',
            ).to(device)

            emb = model(**batch_encoding)
        
        mask = batch_encoding['input_ids'] > 0
        # all_embeddings.append(emb.pooler_output) ## podejscie nr 1 z tokenem CLS
        for i in range(emb[0].size()[0]):
            all_embeddings.append(emb[0][i, mask[i] > 0, :].mean(
                axis=0)[None, :])  # podejscie nr 2 z usrednianiem
    # print(all_embeddings[0].shape)
    return torch.cat(all_embeddings, axis=0).to('cpu')


def _get_embeddings_fasttext(texts, model_name: str, use_cuda: bool) -> torch.Tensor:
    def _pad_tokens(token: torch.Tensor):
        pass
    model = fasttext.load_model(FASTTEXT_MODELS_PATHS[model_name])
    all_embeddings = []
    for batched_text in tqdm(texts):
        with torch.no_grad():
            emb = torch.tensor(model.get_sentence_vector(batched_text), dtype=torch.float32).unsqueeze(0)
        all_embeddings.append(emb)
    # print(all_embeddings[0].shape)
    return torch.cat(all_embeddings, axis=0).to('cpu')


def _create_transfer_embeddings(embeddings, model, use_cuda):
    print("Create transfer learning embeddings...")
    device = torch.device('cuda' if use_cuda else 'cpu')
    new_embeddings = []
    model.to(device)
    for emb in tqdm(embeddings):
        emb = emb.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model({'embeddings': emb})
        new_emb = torch.cat([emb, output], axis=1)
        new_embeddings.append(new_emb)
    return torch.cat(new_embeddings, axis=0).to('cpu')


def create_embeddings(texts, embeddings_path, model_name='xlm-roberta-base', is_transformer: bool = True, use_cuda=True,
                        pickle_embeddings=True, model=None):

    print("Creating embeddings...")
    if is_transformer:
        embeddings = _get_embeddings_transformers(texts, model_name, use_cuda=use_cuda)
    else:
        embeddings = _get_embeddings_fasttext(texts, model_name, use_cuda)

    #print(embeddings.shape)
    #raise None
    print(embeddings.shape)
    if model is not None:
        embeddings = _create_transfer_embeddings(embeddings, model, use_cuda)
        print("Created TRANSFER LEARNING EMBEDDINGS")

    print(embeddings.shape)
    text_idx_to_emb = {}
    for i in range(embeddings.size(0)):
        text_idx_to_emb[i] = embeddings[i].numpy()

    if pickle_embeddings:
        pickle.dump(text_idx_to_emb, open(embeddings_path, 'wb'))

    return text_idx_to_emb