import torch
import torchtext
from torchtext.datasets import text_classification
import os

from TextSentiment import TextSentiment

NGRAMS = 2
DATA_PATH = './.data'

if not os.path.isdir(DATA_PATH):
    os.mkdir(DATA_PATH)

train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](
    root=DATA_PATH,
    ngrams=NGRAMS,
    vocab=None
)

devive = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VOCAB_SIZE = len(train_dataset.get_vocab())
EMBED_DIM = 32
NUM_CLASS = len(train_dataset.get_labels())
model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUM_CLASS).to(devive)
