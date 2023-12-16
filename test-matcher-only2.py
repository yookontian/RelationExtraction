import torch
from models.matcher import ILPMatcher_r_h, HungarianMatcher_r_h, HungarianMatcher, ILPMatcher, HungarianMatcher_dotproduct, ILPMatcher_dotproduct
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
    "pred_rel_logits": torch.rand(2, 10, 2),
    "head_start_logits": torch.rand(2, 10, 5),
    # "head_end_logits": torch.rand(2, 10, 5),
    # "tail_start_logits": torch.rand(2, 10, 5),
    # "tail_end_logits": torch.rand(2, 10, 5),
}

# outputs = {
#     "pred_rel_logits": torch.tensor([[0, 1], [1, 0]], dtype=torch.float64).unsqueeze(0),
#     "head_start_logits":torch.tensor([[0, 1, 0, 0, 0],[0, 0, 0, 1, 0]], dtype=torch.float64).unsqueeze(0),
#     "head_end_logits": torch.tensor([[0, 1, 0, 0, 0],[0, 0, 0, 1, 0]], dtype=torch.float64).unsqueeze(0),
#     "tail_start_logits":torch.tensor([[0, 0, 1, 0, 0],[0, 0, 0, 0, 1]], dtype=torch.float64).unsqueeze(0),
#     "tail_end_logits": torch.tensor([[0, 0, 1, 0, 0],[0, 0, 0, 0, 1]], dtype=torch.float64).unsqueeze(0),
# }



targets = [{
    "relation": torch.tensor([1, 0, 1, 0, 1, 0]),
    "head_start_index": torch.tensor([1, 3, 1, 3, 1, 3]),
    "head_end_index": torch.tensor([1, 3, 2, 4, 3, 1]),
    "tail_start_index": torch.tensor([2, 4, 2, 4, 3, 1]),
    "tail_end_index": torch.tensor([2, 4, 2, 4, 3, 1]),
},
    {
    "relation": torch.tensor([1, 0, 1, 1]),
    "head_start_index": torch.tensor([1, 3, 1, 2]),
    "head_end_index": torch.tensor([1, 3, 1, 2]),
    "tail_start_index": torch.tensor([2, 4, 3, 4]),
    "tail_end_index": torch.tensor([2, 4, 3, 4]),
    }
]

# put the outputs in gpu
# outputs = {k: v.cuda() for k, v in outputs.items()}
# targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

marcher = HungarianMatcher_r_h({
    "relation": torch.tensor([1]),
    "head_entity": torch.tensor([1]),
    "tail_entity": torch.tensor([1])
}, "avg")

marcher_1 = ILPMatcher_r_h({
    "relation": torch.tensor([1]),
    "head_entity": torch.tensor([1]),
    "tail_entity": torch.tensor([1])
}, "avg")

# marcher_1 = ILPMatcher({
#     "relation": torch.tensor([1]),
#     "head_entity": torch.tensor([1]),
#     "tail_entity": torch.tensor([1])
# }, "avg")
#
# marcher_2 = HungarianMatcher_dotproduct({
#     "relation": torch.tensor([1]),
#     "head_entity": torch.tensor([1]),
#     "tail_entity": torch.tensor([1])
# }, "avg")
#
# marcher_3 = ILPMatcher_dotproduct({
#     "relation": torch.tensor([1]),
#     "head_entity": torch.tensor([1]),
#     "tail_entity": torch.tensor([1])
# }, "avg")


# result_1, cost_1 = marcher(outputs, targets)
# print(f"cost_1:\n{cost_1}")
# get the index of the minimum of each row of the cost_1
# cost_1_min_index = torch.argmin(cost_1, dim=-1)
# print(f"cost_1_min_index:\n{cost_1_min_index}")

# result_2, cost_2 = marcher_2(outputs, targets)
# print(f"cost_2:\n{cost_2}")
# cost_2_min_index = torch.argmin(cost_2, dim=-1)
# print(f"cost_2_min_index:\n{cost_2_min_index}")
# a = marcher(outputs, targets)



time_marcher_start = time.time()
print(f"Hungarian, r_h: \n{marcher(outputs, targets)}")
a = marcher(outputs, targets)
time_marcher_end = time.time()
print("time for HungarianMatcher: ", time_marcher_end - time_marcher_start)

time_marcher_start = time.time()
print(f"ILP, r_h: \n{marcher_1(outputs, targets)}")
# a = marcher(outputs, targets)
time_marcher_end = time.time()
print("time for HungarianMatcher: ", time_marcher_end - time_marcher_start)




# for the output, it is a list of tuple, with the first element being the index of the prediction, and the second
# element being the index of the target
# the length of the list is the batch size
# the length of the item in tuple is the minimum of the number of generated triples and the number of gold triples