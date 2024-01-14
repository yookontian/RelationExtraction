import json
from transformers import AutoTokenizer
from transformers import AutoModel
import torch

# load data/NYT/rel.json
with open("data/NYT/rel.json") as f:
    rel = json.load(f)

relation = rel['relation']

# model = "bert-base-cased"
model = "SpanBERT/spanbert-base-cased"
model_ = AutoModel.from_pretrained(model)
tokenizer = AutoTokenizer.from_pretrained(model)
# tokenized_relation = [tokenizer.encode(r, add_special_tokens=False)[0] for r in relation]

tokenized_relation = [tokenizer.encode(r, return_tensors="pt",) for r in relation]

# convert tokenized_relation to tensor
tokenized_relation = torch.cat(tokenized_relation, dim=0)

embeded_rel = model_(tokenized_relation)['pooler_output']
# save the embeded_rel to data/NYT/embeded_rel.pt
torch.save(embeded_rel, "data/NYT/embeded_rel-SpanBERT.pt")
print("end")