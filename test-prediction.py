import torch
import random
import numpy as np
import os
from utils.data import build_data
from utils.data import load_data_setting
from trainer.trainer import Trainer
from models.setpred4RE import SetPred4RE
from transformers import AutoTokenizer


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class make_args:
    def __init__(self):
        self.generated_data_directory = "data/NYT/generated_data/"
        self.generated_param_directory = "data/NYT/Hungarian-model_param-bi_regressive_decoder_1_2layer-class_embed-SpanBERT-fix_logits-epoch201-300/"
        self.dataset_name = "NYT"
        self.model_name = "Hungarian-model_param-bi_regressive_decoder_1_2layer-class_embed-SpanBERT-fix_logits-epoch201-300"
        # self.bert_directory = "bert-base-cased"
        self.bert_directory = "SpanBERT/spanbert-base-cased"
        self.train_file = "data/NYT/exact_data/train.json"
        self.valid_file = "data/NYT/exact_data/valid.json"
        self.test_file = "data/NYT/exact_data/test.json"
        self.num_generated_triples = 15
        # self.num_generated_triples = 25

        self.num_decoder_layers = 2

        self.matcher = "avg"
        self.rel_loss_weight = 1.0
        self.head_ent_loss_weight = 2.0
        self.tail_ent_loss_weight = 2.0
        self.fix_bert_embeddings = True
        self.batch_size = 8
        self.max_epoch = 100
        self.gradient_accumulation_steps = 1
        self.decoder_lr = 2e-5
        # self.decoder_lr = 2e-6  #201-300
        self.encoder_lr = 1e-5
        self.lr_decay = 0.01
        self.weight_decay = 1e-5
        self.max_grad_norm = 2.5
        self.optimizer = "AdamW"
        self.na_rel_coef = 0.5
        # self.na_rel_coef = 0.8

        # Evaluation arguments
        self.n_best_size = 100
        self.max_span_length = 12

        # Misc arguments
        self.refresh = False
        self.use_gpu = True
        self.visible_gpu = 1
        self.random_seed = 1

        # new attribute
        self.use_ILP = False
        self.use_dotproduct = False
        self.use_regressive_decoder = True
        self.batch_size = 8

    def __iter__(self):
        for attr in dir(self):
            if not callable(getattr(self, attr)) and not attr.startswith("__"):
                yield attr

a = make_args()


data = load_data_setting(a)
print(data.relational_alphabet.instances)
# stop the programe
exit()

model = SetPred4RE(a, data.relational_alphabet.size())

# print(model)
model.load_state_dict(torch.load("data/NYT/Hungarian-model_param-bi_regressive_decoder_1_2layer-class_embed-SpanBERT-epoch101-200/ Hungarian-model_param-bi_regressive_decoder_1_2layer-class_embed-SpanBERT-epoch101-200_NYT_epoch_84_f1_0.9305.model")['state_dict'])
# model.load_state_dict(torch.load("data/NYT/Hungarian-model_param-bi_regressive_decoder_2layer-class_embed-SpanBERT/ Hungarian-model_param-bi_regressive_decoder_2layer-class_embed-SpanBERT_NYT_epoch_99_f1_0.9291.model")['state_dict'])
# model.load_state_dict(torch.load("data/NYT/Hungarian-model-param/Hungarian-model-param HungarianModel_NYT_epoch_95_f1_0.9244.model")['state_dict'])
# model.load_state_dict(torch.load("data/NYT/ILP-model_param-bi_regressive_decoder_1_2layer-class_embed-SpanBERT/ ILP-model_param-bi_regressive_decoder_1_2layer-class_embed-SpanBERT_NYT_epoch_94_f1_0.9134.model")['state_dict'])

"""
trainer = Trainer(model, data, a)
print(f"batch_size: {trainer.args.batch_size}")
"""

tokenizer = AutoTokenizer.from_pretrained("SpanBERT/spanbert-base-cased")

model.eval()
with torch.no_grad():
    t = "I don't think the Yunnan government or any other organization has dominion over the jungles of Xishuangbanna ."
    tokenized_t = tokenizer(t, return_tensors="pt", padding=True)
    print(tokenized_t)
    gen_triples = model.gen_triples(tokenized_t["input_ids"], tokenized_t["attention_mask"], {"seq_len": [len(tokenized_t["input_ids"][0])], "sent_idx": [0]})

    # print(gen_triples)
    rel = []
    head_span = []
    tail_span = []
    for (_, item) in gen_triples.items():
        rel.append(data.relational_alphabet.instances[item[0].pred_rel])
        head_span.append(tokenizer.decode(tokenized_t["input_ids"][0][item[0].head_start_index: item[0].head_end_index + 1]))
        tail_span.append(tokenizer.decode(tokenized_t["input_ids"][0][item[0].tail_start_index: item[0].tail_end_index + 1]))

    # show all attributes of data.relational_alphabet
    for r, h, t in zip(rel, head_span, tail_span):
        print(f"rel: {r}, head_span: {h}, tail_span: {t}")

    print("end")

    """    
    result = trainer.eval_model(trainer.data.test_loader)
    print(result)
    """

    # sentence = ["A French court sentenced six Algerian-French men to prison terms of up to 10 years on Tuesday for their role in a 2001 plot to attack the United States Embassy in Paris , closing the books on one of France 's most serious terrorist cases .",
    #             ""]
    # input_ids, attention_mask = tokenizer(sentence, return_tensors="pt" ,padding=True).input_ids, tokenizer(sentence, return_tensors="pt", padding=True).attention_mask
    # for token in input_ids[0]:
        # print(token.item(), ": ", tokenizer.convert_ids_to_tokens(token.item()))
    # input_ids = input_ids.cuda()
    # attention_mask = attention_mask.cuda()
    # print(input_ids.shape)
    # print(attention_mask.shape)
    # info = {"seq_len": [len(input_ids[0]), len(input_ids[1])], "sent_idx": [0, 1]}
    # # result = model(input_ids, attention_mask)
    # print(result.keys())
    # print(result['pred_rel_logits'].shape)
    # relation = result['pred_rel_logits'][0].argmax(-1)
    # print(relation)
    # result = model.gen_triples(input_ids, attention_mask, info)
    # head_span = []
    # tail_span = []
    # for num_tri in range(len(result[0])):
    #
    #     head_tokens = []
    #     for idx in range(result[0][num_tri].head_start_index, result[0][num_tri].head_end_index + 1):
    #
    #         head_tokens.append(input_ids[0][idx].item())
    #
    #     head_span.append(tokenizer.decode(head_tokens))
    #
    #     tail_tokens = []
    #     for idx in range(result[0][num_tri].tail_start_index, result[0][num_tri].tail_end_index + 1):
    #         tail_tokens.append(input_ids[0][idx].item())
    #
    #     tail_span.append(tokenizer.decode(tail_tokens))
    #
    #
    # print(f"head_span: {head_span}")
    # print(f"tail_span: {tail_span}")

# print the number of trainable parameters
# print(f"number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")