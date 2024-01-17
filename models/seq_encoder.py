import torch.nn as nn
from transformers import BertModel, BertTokenizer, AutoModel


class SeqEncoder(nn.Module):
    def __init__(self, args):
        super(SeqEncoder, self).__init__()
        self.args = args
        if args.bert_directory == "bert-base-cased":
            self.bert = BertModel.from_pretrained(args.bert_directory)
        else:
            self.bert = AutoModel.from_pretrained(args.bert_directory)
        if args.fix_bert_embeddings:
            self.bert.embeddings.word_embeddings.weight.requires_grad = False
            self.bert.embeddings.position_embeddings.weight.requires_grad = False
            self.bert.embeddings.token_type_embeddings.weight.requires_grad = False
        self.config = self.bert.config

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        pooler_output = output.pooler_output
        return last_hidden_state, pooler_output