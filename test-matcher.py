import torch
from models.matcher import HungarianMatcher, ILPMatcher, HungarianMatcher_dotproduct, ILPMatcher_dotproduct
import time

seed_value = 46
torch.manual_seed(seed_value)
# for the relation, the output is in dimension [batch_size, num_generated_triples, num_classes]
# for the start and end logits, the dimension is [batch_size, num_generated_triples, seq_len]
# outputs = {
#     "pred_rel_logits": torch.rand(2, 3, 2)[0].unsqueeze(0),
#     "head_start_logits": torch.rand(2, 3, 5)[0].unsqueeze(0),
#     "head_end_logits": torch.rand(2, 3, 5)[0].unsqueeze(0),
#     "tail_start_logits": torch.rand(2, 3, 5)[0].unsqueeze(0),
#     "tail_end_logits": torch.rand(2, 3, 5)[0].unsqueeze(0),
# }

outputs = {
    "pred_rel_logits": torch.rand(2, 2, 2),
    "head_start_logits": torch.rand(2, 2, 5),
    "head_end_logits": torch.rand(2, 2, 5),
    "tail_start_logits": torch.rand(2, 2, 5),
    "tail_end_logits": torch.rand(2, 2, 5),
}

# outputs = {
#     "pred_rel_logits": torch.tensor([[0, 1], [1, 0]], dtype=torch.float64).unsqueeze(0),
#     "head_start_logits":torch.tensor([[0, 1, 0, 0, 0],[0, 0, 0, 1, 0]], dtype=torch.float64).unsqueeze(0),
#     "head_end_logits": torch.tensor([[0, 1, 0, 0, 0],[0, 0, 0, 1, 0]], dtype=torch.float64).unsqueeze(0),
#     "tail_start_logits":torch.tensor([[0, 0, 1, 0, 0],[0, 0, 0, 0, 1]], dtype=torch.float64).unsqueeze(0),
#     "tail_end_logits": torch.tensor([[0, 0, 1, 0, 0],[0, 0, 0, 0, 1]], dtype=torch.float64).unsqueeze(0),
# }



targets = [{
    "relation": torch.tensor([1, 0]),
    "head_start_index": torch.tensor([1, 3]),
    "head_end_index": torch.tensor([1, 3]),
    "tail_start_index": torch.tensor([2, 4]),
    "tail_end_index": torch.tensor([2, 4]),
},
    {
    "relation": torch.tensor([1, 0, 1, 1]),
    "head_start_index": torch.tensor([1, 3, 1, 2]),
    "head_end_index": torch.tensor([1, 3, 1, 2]),
    "tail_start_index": torch.tensor([2, 4, 3, 4]),
    "tail_end_index": torch.tensor([2, 4, 3, 4]),
    }
]

marcher = HungarianMatcher({
    "relation": torch.tensor([1]),
    "head_entity": torch.tensor([1]),
    "tail_entity": torch.tensor([1])
}, "avg")

marcher_1 = ILPMatcher({
    "relation": torch.tensor([1]),
    "head_entity": torch.tensor([1]),
    "tail_entity": torch.tensor([1])
}, "avg")

marcher_2 = HungarianMatcher_dotproduct({
    "relation": torch.tensor([1]),
    "head_entity": torch.tensor([1]),
    "tail_entity": torch.tensor([1])
}, "avg")

marcher_3 = ILPMatcher_dotproduct({
    "relation": torch.tensor([1]),
    "head_entity": torch.tensor([1]),
    "tail_entity": torch.tensor([1])
}, "avg")

# time_marcher_start = time.time()
print(f"Hungarian: \n{marcher(outputs, targets)}")
# time_marcher_end = time.time()
# print("time for HungarianMatcher: ", time_marcher_end - time_marcher_start)

# time_marcher_start = time.time()
print(f"Hungarian_dotproduct: \n{marcher_2(outputs, targets)}")
# time_marcher_end = time.time()
# print("time for HungarianMatcher_dotproduct: ", time_marcher_end - time_marcher_start)

# time_marcher_start = time.time()
print(f"ILP: \n{marcher_1(outputs, targets)}")
# time_marcher_end = time.time()
# print("time for ILPMatcher: ", time_marcher_end - time_marcher_start)


# time_marcher_start = time.time()
print(f"ILP_dotproduct: \n{marcher_3(outputs, targets)}")
# time_marcher_end = time.time()
# print("time for ILPMatcher_dotproduct: ", time_marcher_end - time_marcher_start)


# for the output, it is a list of tuple, with the first element being the index of the prediction, and the second
# element being the index of the target
# the length of the list is the batch size
# the length of the item in tuple is the minimum of the number of generated triples and the number of gold triples