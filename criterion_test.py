from models.seq_encoder import SeqEncoder
from models.set_decoder import SetDecoder
import torch
from transformers import BertTokenizer
from models.set_criterion import SetCriterion

import time

start = time.time()

class Args:
    def __init__(self, **kwargs):
        self.bert_directory = "bert-base-cased"
        self.fix_bert_embeddings = True


bert_args = Args()

encoder = SeqEncoder(bert_args)

tokenizer = BertTokenizer.from_pretrained(bert_args.bert_directory)


text = ["Hello, my dog is cute", "Do you want to play with me? I think we have some commons."]
tokenized_text = tokenizer(text, return_tensors="pt", padding=True)
last_hidden_state, pooler_output = encoder(input_ids=tokenized_text["input_ids"], attention_mask=tokenized_text["attention_mask"])

print("last_hidden_state: ", last_hidden_state.shape)
print("pooler_output: ", pooler_output.shape, "\n====================\n")


decoder = SetDecoder(encoder.config, 2, 2, 9, return_intermediate=False, use_ILP=True)
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

# print("decoder_output: \n", len(decoder_output))
outputs = {'pred_rel_logits': class_logits, 'head_start_logits': head_start_logits, 'head_end_logits': head_end_logits, 'tail_start_logits': tail_start_logits, 'tail_end_logits': tail_end_logits}
targets = [{
    "relation": torch.tensor([1, 2]),
    "head_start_index": torch.tensor([3, 2]),
    "head_end_index": torch.tensor([4, 5]),
    "tail_start_index": torch.tensor([8, 3]),
    "tail_end_index": torch.tensor([9, 3]),
},
    {
    "relation": torch.tensor([1, 8, 1]),
    "head_start_index": torch.tensor([7, 3, 1]),
    "head_end_index": torch.tensor([9, 3, 1]),
    "tail_start_index": torch.tensor([14, 3, 1]),
    "tail_end_index": torch.tensor([15, 3, 1]),
    }
]

criterion = SetCriterion(9,
{
    "relation": torch.tensor([1]),
    "head_entity": torch.tensor([1]),
    "tail_entity": torch.tensor([1])
}
, na_coef=0.1, losses=["entity", "relation"], matcher="avg", use_ILP=True)

# print(f"output shape: {outputs['pred_rel_logits'].shape}")
# print(f"targets shape: {len(targets)}")
#
indices = criterion.matcher(outputs, targets)
print("indices of criterion: ", indices)
#
# entity_losses = criterion.get_loss("entity", outputs, targets, indices)
# print("entity loss: ", entity_losses)

relation_losses = criterion.get_loss("relation", outputs, targets, indices)
print("reloation loss: ", relation_losses)


# loss = criterion(outputs, targets)
# print("loss in test: ", loss)

end = time.time()
print("time: ", end - start)