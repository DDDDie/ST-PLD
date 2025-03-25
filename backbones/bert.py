from operator import mod
import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertPreTrainedModel, BertModel,  AutoModelForMaskedLM, BertForMaskedLM
from torch.nn.parameter import Parameter
from .utils import PairEnum
from sentence_transformers import SentenceTransformer
from losses import SupConLoss

activation_map = {'relu': nn.ReLU(), 'tanh': nn.Tanh()}
       
class BERT_STPLD(BertPreTrainedModel):
    
    def __init__(self, config, args):

        super(BERT_STPLD, self).__init__(config)
        self.num_labels = args.num_labels
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = activation_map[args.activation]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.args = args
        # new
        if args.pretrain or (not args.wo_self):
        # if args.pretrain:
            self.classifier = nn.Linear(config.hidden_size, args.num_labels)
                
        self.mlp_head = nn.Linear(config.hidden_size, args.num_labels)
            
        self.init_weights()

    def forward(self, input_ids = None, token_type_ids = None, attention_mask=None , feature_ext = False):

        outputs = self.bert(
            input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_hidden_states=True)

        encoded_layer_12 = outputs.hidden_states
        last_output_tokens = encoded_layer_12[-1]     
        features = last_output_tokens.mean(dim = 1)
            
        features = self.dense(features)
        pooled_output = self.activation(features)   
        pooled_output = self.dropout(features)
        
        if self.args.pretrain or (not self.args.wo_self):
            logits = self.classifier(pooled_output)
            
        mlp_outputs = self.mlp_head(pooled_output)
        
        if feature_ext:
            if self.args.pretrain or (not self.args.wo_self):
                return features, logits
            else:
                return features, mlp_outputs

        else:
            if self.args.pretrain or (not self.args.wo_self):
                return mlp_outputs, logits
            else:
                return mlp_outputs, mlp_outputs