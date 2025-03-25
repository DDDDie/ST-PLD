from .bert import BERT_STPLD
from .glove import GloVeEmbeddingVectorizer
from .sae import get_stacked_autoencoder

backbones_map = {   
                    'bert_STPLD': BERT_STPLD,
                    'glove': GloVeEmbeddingVectorizer,
                    'sae': get_stacked_autoencoder,
                }