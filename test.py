import torch
from models.matcher import HungarianMatcher

# have a tensor with dim [1,2,3], with the value random between 0 and 1


outputs = {
    "pred_rel_logits": torch.rand(1, 4, 3),
    "head_start_logits": torch.rand(1, 4, 20),
    "head_end_logits": torch.rand(1, 4, 20),
    "tail_start_logits": torch.rand(1, 4, 20),
    "tail_end_logits": torch.rand(1, 4, 320),
}


targets = [{
    "relation": torch.tensor([1, 2]),
    "head_start_index": torch.tensor([3, 7]),
    "head_end_index": torch.tensor([4, 9]),
    "tail_start_index": torch.tensor([8, 14]),
    "tail_end_index": torch.tensor([9, 15]),
}]

marcher = HungarianMatcher({
    "relation": torch.tensor([0.3]),
    "head_entity": torch.tensor([0.3]),
    "tail_entity": torch.tensor([0.3])
}, "avg")


print(marcher(outputs, targets))
