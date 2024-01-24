import torch
import random
import numpy as np
import os
from utils.data import build_data
from utils.data import load_data_setting
from trainer.trainer import Trainer
from models.setpred4RE import SetPred4RE
import wandb


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
        # self.generated_data_directory = "data/NYT/RoBERTa_data/"
        self.generated_param_directory = "data/NYT/Hungarian-model-param-regressive-2layer-SpanBERT_class_embed-PosEmbed-SpanBERT/"
        self.dataset_name = "NYT"
        self.model_name = "Hungarian-model-param-regressive-2layer-SpanBERT_class_embed-PosEmbed-SpanBERT"
        # self.bert_directory = "bert-base-cased"
        self.bert_directory = "SpanBERT/spanbert-base-cased"
        # self.bert_directory = "roberta-base"
        self.train_file = "data/NYT/exact_data/train.json"
        # self.valid_file = "data/NYT/exact_data/valid.json"
        self.valid_file = "data/NYT/exact_data/test.json"
        # self.test_file = "data/NYT/exact_data/test.json"
        # self.num_generated_triples = 15
        self.num_generated_triples = 10

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
        self.none_class = True  # defult is True
        self.positional_embedding = True

    def __iter__(self):
        for attr in dir(self):
            if not callable(getattr(self, attr)) and not attr.startswith("__"):
                yield attr

a = make_args()


data = load_data_setting(a)


model = SetPred4RE(a, data.relational_alphabet.size())

# model.load_state_dict(torch.load("data/NYT/Hungarian-model_param-bi_regressive_decoder_1_2layer-class_embed-SpanBERT-fix_logits-epoch101-200/ Hungarian-model_param-bi_regressive_decoder_1_2layer-class_embed-SpanBERT-fix_logits-epoch101-200_NYT_epoch_97_f1_0.9314.model")['state_dict'])

# fix the logits layers af 100 epochs
# model.decoder.head_start_metric_3.weight.requires_grad = False
# model.decoder.head_end_metric_3.weight.requires_grad = False
# model.decoder.tail_start_metric_3.weight.requires_grad = False
# model.decoder.tail_end_metric_3_back.weight.requires_grad = False
# model.decoder.tail_start_metric_3_back.weight.requires_grad = False
# model.decoder.head_end_metric_3_back.weight.requires_grad = False




# start a new wandb run to track this script
wandb.init(
    project="SPN4RE",
    name="SPN4RE-Hungarian-model-param-regressive-2layer-SpanBERT_class_embed-PosEmbed-SpanBERT)",
)

wandb.watch(model, log_freq=100)
trainer = Trainer(model, data, a)

print(f"batch_size: {trainer.args.batch_size}")
print("start training")
# with torch.no_grad():
trainer.train_model()

# wandb.finish()

