from models.seq_encoder import SeqEncoder
from models.set_decoder import SetDecoder
from models.set_regressive_decoder import SetRegressiveDecoder
import torch
from transformers import BertTokenizer

class Args:
    def __init__(self, **kwargs):
        # self.bert_directory = "bert-base-cased"
        self.bert_directory = "SpanBERT/spanbert-base-cased"
        self.fix_bert_embeddings = True



bert_args = Args()

encoder = SeqEncoder(bert_args)

tokenizer = BertTokenizer.from_pretrained(bert_args.bert_directory)


text = ["Hello, my dog is cute", "Do you want to play with me? I think we have some commons."]
tokenized_text = tokenizer(text, return_tensors="pt", padding=True)
last_hidden_state, pooler_output = encoder(input_ids=tokenized_text["input_ids"], attention_mask=tokenized_text["attention_mask"])

print("last_hidden_state: ", last_hidden_state.shape)
print("pooler_output: ", pooler_output.shape, "\n====================\n")


decoder = SetRegressiveDecoder(encoder.config, 5, 2, 24, return_intermediate=False)
# decoder = SetDecoder(encoder.config, 5, 2, 10, return_intermediate=False)
decoder_output = decoder(encoder_hidden_states=last_hidden_state, encoder_attention_mask=tokenized_text["attention_mask"])
class_logits, head_start_logits, head_end_logits, tail_start_logits, tail_end_logits  = decoder_output

print("class_logits: ", class_logits.shape)
print("head_start_logits: ", head_start_logits.shape)
print("head_end_logits: ", head_end_logits.shape)
print("tail_start_logits: ", tail_start_logits.shape)
print("tail_end_logits: ", tail_end_logits.shape)

# have the argmax of the class_logits
class_logits_argmax = class_logits.argmax(-1)
print("class_logits_argmax: ", class_logits_argmax)