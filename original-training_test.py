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
        self.generated_param_directory = "data/NYT/Hungarian-model-spanbert-regressive/"
        self.dataset_name = "NYT"
        self.model_name = "HungarianModel"
        # self.bert_directory = "bert-base-cased"
        self.bert_directory = "SpanBERT/spanbert-base-cased"
        self.train_file = "data/NYT/exact_data/train.json"
        self.valid_file = "data/NYT/exact_data/valid.json"
        self.test_file = "data/NYT/exact_data/test.json"
        self.num_generated_triples = 15
        self.num_decoder_layers = 3
        self.na_rel_coef = 0.25
        self.matcher = "avg"
        self.rel_loss_weight = 1.0
        self.head_ent_loss_weight = 2.0
        self.tail_ent_loss_weight = 2.0
        self.fix_bert_embeddings = True
        self.batch_size = 8
        self.max_epoch = 100
        self.gradient_accumulation_steps = 1
        self.decoder_lr = 2e-5
        self.encoder_lr = 1e-5
        self.lr_decay = 0.01
        self.weight_decay = 1e-5
        self.max_grad_norm = 2.5
        self.optimizer = "AdamW"

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
        self.batch_size = 4

    def __iter__(self):
        for attr in dir(self):
            if not callable(getattr(self, attr)) and not attr.startswith("__"):
                yield attr

a = make_args()


data = load_data_setting(a)


model = SetPred4RE(a, data.relational_alphabet.size())



# start a new wandb run to track this script
wandb.init(
    project="SPN4RE",
    name="SPN4RE-NYT-Hungarian-SpanBert-regressive-0.25coef",
)

wandb.watch(model, log_freq=100)

trainer = Trainer(model, data, a)
print(f"batch_size: {trainer.args.batch_size}")
print("start training")
# with torch.no_grad():
trainer.train_model()

wandb.finish()