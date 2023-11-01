"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np

import time

from pyscipopt import Model, quicksum


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, loss_weight, matcher):
        super().__init__()
        self.cost_relation = loss_weight["relation"]
        self.cost_head = loss_weight["head_entity"]
        self.cost_tail = loss_weight["tail_entity"]
        self.matcher = matcher

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_rel_logits": Tensor of dim [batch_size, num_generated_triples, num_classes] with the classification logits
                 "{head, tail}_{start, end}_logits": Tensor of dim [batch_size, num_generated_triples, seq_len] with the predicted index logits
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_generated_triples, num_gold_triples)
        """
        # hungarian_start = time.time()
        bsz, num_generated_triples = outputs["pred_rel_logits"].shape[:2]
        # We flatten to compute the cost matrices in a batch
        pred_rel = outputs["pred_rel_logits"].flatten(0, 1).softmax(-1)  # [bsz * num_generated_triples, num_classes]
        gold_rel = torch.cat([v["relation"] for v in targets])
        # print("pred_rel: \n", pred_rel)
        # print("gold_rel:\n ", gold_rel)
        # after masking the pad token
        # print(f'outputs["head_start_logits"]: \n{outputs["head_start_logits"]}')
        # replace all the nan value with -1e9 in the tensor outputs["head_start_logits"]
        outputs["head_start_logits"] = torch.where(torch.isnan(outputs["head_start_logits"]), torch.full_like(outputs["head_start_logits"], -1e9), outputs["head_start_logits"])
        pred_head_start = outputs["head_start_logits"].flatten(0, 1).softmax(-1)  # [bsz * num_generated_triples, seq_len]
        outputs["head_end_logits"] = torch.where(torch.isnan(outputs["head_end_logits"]), torch.full_like(outputs["head_end_logits"], -1e9), outputs["head_end_logits"])
        pred_head_end = outputs["head_end_logits"].flatten(0, 1).softmax(-1)
        outputs["tail_start_logits"] = torch.where(torch.isnan(outputs["tail_start_logits"]), torch.full_like(outputs["tail_start_logits"], -1e9), outputs["tail_start_logits"])
        pred_tail_start = outputs["tail_start_logits"].flatten(0, 1).softmax(-1)
        outputs["tail_end_logits"] = torch.where(torch.isnan(outputs["tail_end_logits"]), torch.full_like(outputs["tail_end_logits"], -1e9), outputs["tail_end_logits"])
        pred_tail_end = outputs["tail_end_logits"].flatten(0, 1).softmax(-1)

        gold_head_start = torch.cat([v["head_start_index"] for v in targets])
        gold_head_end = torch.cat([v["head_end_index"] for v in targets])
        gold_tail_start = torch.cat([v["tail_start_index"] for v in targets])
        gold_tail_end = torch.cat([v["tail_end_index"] for v in targets])

        # print(f'====matcher\npred_head_start"]:\n{pred_head_start}')

        if self.matcher == "avg":
            # hungarian_start = time.time()
            cost = - self.cost_relation * pred_rel[:, gold_rel] - self.cost_head * 1/2 * (pred_head_start[:, gold_head_start] + pred_head_end[:, gold_head_end]) - self.cost_tail * 1/2 * (pred_tail_start[:, gold_tail_start] + pred_tail_end[:, gold_tail_end])
            # hungarian_end = time.time()
            # print("time for hungarian: ", hungarian_end - hungarian_start)
            # print(f"cost: \n{cost}")

        elif self.matcher == "min":
            cost = torch.cat([pred_head_start[:, gold_head_start].unsqueeze(1), pred_rel[:, gold_rel].unsqueeze(1), pred_head_end[:, gold_head_end].unsqueeze(1), pred_tail_start[:, gold_tail_start].unsqueeze(1), pred_tail_end[:, gold_tail_end].unsqueeze(1)], dim=1)
            cost = - torch.min(cost, dim=1)[0]
        else:
            raise ValueError("Wrong matcher")

        # print("1. the shape of cost: ", cost.shape)

        # print("cost: \n", cost)

        cost = cost.view(bsz, num_generated_triples, -1).cpu()

        # print("2. the shape of cost: ", cost.shape)

        num_gold_triples = [len(v["relation"]) for v in targets]

        pair = []
        for i, c in enumerate(cost.split(num_gold_triples, -1)):
            n = num_gold_triples[i]
            # print(f"c[{i}]:\n{c[i]}")
            k = len(c[i])
            if n > k:
                print("n_class: ", n, "n_generated_triple: ", k)
            pair.append(linear_sum_assignment(c[i]))

        # hungarian_end = time.time()
        # print("time for hungarian: ", hungarian_end - hungarian_start)
        # indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost.split(num_gold_triples, -1))]


        # i: row, j: col, if bsz > 1 [(tuple for batch 1), (tuple for batch 2), ...]
        # return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in pair]


def get_dot_cost(pred_rel, gold_rel):
    gold_rel_logits = torch.zeros(gold_rel.shape[-1], pred_rel.shape[-1])
    for i in range(gold_rel.shape[-1]):
        # Set the column corresponding to the value in 'b' to 1
        gold_rel_logits[i, gold_rel[i]] = 1.0

    pred_rel_norm = nn.functional.normalize(pred_rel, p=2, dim=1)
    gold_rel_logits_norm = nn.functional.normalize(gold_rel_logits, p=2, dim=1).to(pred_rel.device)
    cost_rel = torch.mm(pred_rel_norm, gold_rel_logits_norm.t())
    return cost_rel



class HungarianMatcher_dotproduct(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, loss_weight, matcher):
        super().__init__()
        self.cost_relation = loss_weight["relation"]
        self.cost_head = loss_weight["head_entity"]
        self.cost_tail = loss_weight["tail_entity"]
        self.matcher = matcher

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_rel_logits": Tensor of dim [batch_size, num_generated_triples, num_classes] with the classification logits
                 "{head, tail}_{start, end}_logits": Tensor of dim [batch_size, num_generated_triples, seq_len] with the predicted index logits
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_generated_triples, num_gold_triples)
        """
        # time_hungarian_dot_start = time.time()

        bsz, num_generated_triples = outputs["pred_rel_logits"].shape[:2]
        # We flatten to compute the cost matrices in a batch
        pred_rel = outputs["pred_rel_logits"].flatten(0, 1).softmax(-1)  # [bsz * num_generated_triples, num_classes]
        gold_rel = torch.cat([v["relation"] for v in targets])
        # print("pred_rel: \n", pred_rel)
        # print("gold_rel:\n ", gold_rel)
        # after masking the pad token
        # replace all the nan value with -1e9 in the tensor outputs["head_start_logits"]
        outputs["head_start_logits"] = torch.where(torch.isnan(outputs["head_start_logits"]), torch.full_like(outputs["head_start_logits"], -1e9), outputs["head_start_logits"])
        pred_head_start = outputs["head_start_logits"].flatten(0, 1).softmax(-1)  # [bsz * num_generated_triples, seq_len]
        outputs["head_end_logits"] = torch.where(torch.isnan(outputs["head_end_logits"]), torch.full_like(outputs["head_end_logits"], -1e9), outputs["head_end_logits"])
        pred_head_end = outputs["head_end_logits"].flatten(0, 1).softmax(-1)
        outputs["tail_start_logits"] = torch.where(torch.isnan(outputs["tail_start_logits"]), torch.full_like(outputs["tail_start_logits"], -1e9), outputs["tail_start_logits"])
        pred_tail_start = outputs["tail_start_logits"].flatten(0, 1).softmax(-1)
        outputs["tail_end_logits"] = torch.where(torch.isnan(outputs["tail_end_logits"]), torch.full_like(outputs["tail_end_logits"], -1e9), outputs["tail_end_logits"])
        pred_tail_end = outputs["tail_end_logits"].flatten(0, 1).softmax(-1)


        gold_head_start = torch.cat([v["head_start_index"] for v in targets])
        gold_head_end = torch.cat([v["head_end_index"] for v in targets])
        gold_tail_start = torch.cat([v["tail_start_index"] for v in targets])
        gold_tail_end = torch.cat([v["tail_end_index"] for v in targets])

        # print(pred_rel)
        # print(gold_rel)


        # print("pred_rel: \n", pred_rel.shape)
        # print("pred_rel[:, gold_rel]: ", pred_rel[:, gold_rel].shape)
        # print("pred_head_start[:, gold_head_start]: ", pred_head_start[:, gold_head_start].shape)
        if self.matcher == "avg":
            # time_hungarian_dot_start = time.time()
            cost_rel = get_dot_cost(pred_rel, gold_rel)
            cost_head_start = get_dot_cost(pred_head_start, gold_head_start)
            cost_head_end = get_dot_cost(pred_head_end, gold_head_end)
            cost_tail_start = get_dot_cost(pred_tail_start, gold_tail_start)
            cost_tail_end = get_dot_cost(pred_tail_end, gold_tail_end)

            cost = - self.cost_relation * cost_rel - self.cost_head * 1/2 * (cost_head_start + cost_head_end) - self.cost_tail * 1/2 * (cost_tail_start + cost_tail_end)

            # time_hungarian_dot_end = time.time()
            # print("time for hungarian_dot: ", time_hungarian_dot_end - time_hungarian_dot_start)
        elif self.matcher == "min":
            pass
        else:
            raise ValueError("Wrong matcher")

        # print("1. the shape of cost: ", cost.shape)
        # print("cost: \n", cost)

        cost = cost.view(bsz, num_generated_triples, -1).cpu()

        # print("2. the shape of cost: ", cost.shape)

        num_gold_triples = [len(v["relation"]) for v in targets]

        # linear_sum_assignment(cost_matrix: array, maximaze: bool = False) -> array (row_ind, col_ind)
        # print("num_gold_triples: ", num_gold_triples)
        # print("cost.split(num_gold_triples, -1) \n", cost.split(num_gold_triples, -1))

        # print("cost: \n", cost)
        # if bsz > 1, num_gold_triples is a list,
        # first split out [bsz, num_generated_triples, num_gold_triples for batch-i], but only take c[i], i is the batch
        # for i, c in enumerate(cost.split(num_gold_triples, -1)):
        #     print("i: ", i)
        #     print("c[i] shape: ", c[i].shape)
        #     print("c[i]: ", c[i])
        #     print(f"linear_sum_assignment(c[i]): \n{linear_sum_assignment(c[i])}")

        pair = []
        for i, c in enumerate(cost.split(num_goldILP-model_param-noNoneClass-NoABS-dotproduct_triples, -1)):
            n = num_gold_triples[i]
            # print(f"(dot_production) c[{i}]:\n{c[i]}")
            k = len(c[i])
            if n > k:
                print("n_class: ", n, "n_generated_triple: ", k)
            pair.append(linear_sum_assignment(c[i]))

        # time_hungarian_dot_end = time.time()
        # print("time for hungarian_dot: ", time_hungarian_dot_end - time_hungarian_dot_start)
        # indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost.split(num_gold_triples, -1))]


        # i: row, j: col, if bsz > 1 [(tuple for batch 1), (tuple for batch 2), ...]
        # return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in pair]





class ILPMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, loss_weight, matcher):
        super().__init__()
        self.cost_relation = loss_weight["relation"]
        self.cost_head = loss_weight["head_entity"]
        self.cost_tail = loss_weight["tail_entity"]
        self.matcher = matcher

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_rel_logits": Tensor of dim [batch_size, num_generated_triples, num_classes] with the classification logits
                 "{head, tail}_{start, end}_logits": Tensor of dim [batch_size, num_generated_triples, seq_len] with the predicted index logits
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_generated_triples, num_gold_triples)
        """
        bsz, num_generated_triples = outputs["pred_rel_logits"].shape[:2]
        # We flatten to compute the cost matrices in a batch
        pred_rel = outputs["pred_rel_logits"].flatten(0, 1).softmax(-1)  # [bsz * num_generated_triples, num_classes]
        gold_rel = torch.cat([v["relation"] for v in targets])
        # print("pred_rel: \n", pred_rel)
        # print("gold_rel:\n ", gold_rel)
        # after masking the pad token
        # replace all the nan value with -1e9 in the tensor outputs["head_start_logits"]
        outputs["head_start_logits"] = torch.where(torch.isnan(outputs["head_start_logits"]), torch.full_like(outputs["head_start_logits"], -1e9), outputs["head_start_logits"])
        pred_head_start = outputs["head_start_logits"].flatten(0, 1).softmax(-1)  # [bsz * num_generated_triples, seq_len]
        outputs["head_end_logits"] = torch.where(torch.isnan(outputs["head_end_logits"]), torch.full_like(outputs["head_end_logits"], -1e9), outputs["head_end_logits"])
        pred_head_end = outputs["head_end_logits"].flatten(0, 1).softmax(-1)
        outputs["tail_start_logits"] = torch.where(torch.isnan(outputs["tail_start_logits"]), torch.full_like(outputs["tail_start_logits"], -1e9), outputs["tail_start_logits"])
        pred_tail_start = outputs["tail_start_logits"].flatten(0, 1).softmax(-1)
        outputs["tail_end_logits"] = torch.where(torch.isnan(outputs["tail_end_logits"]), torch.full_like(outputs["tail_end_logits"], -1e9), outputs["tail_end_logits"])
        pred_tail_end = outputs["tail_end_logits"].flatten(0, 1).softmax(-1)

        gold_head_start = torch.cat([v["head_start_index"] for v in targets])
        gold_head_end = torch.cat([v["head_end_index"] for v in targets])
        gold_tail_start = torch.cat([v["tail_start_index"] for v in targets])
        gold_tail_end = torch.cat([v["tail_end_index"] for v in targets])
        # print("pred_rel: \n", pred_rel.shape)
        # print("pred_rel[:, gold_rel]: ", pred_rel[:, gold_rel].shape)
        # print("pred_head_start[:, gold_head_start]: ", pred_head_start[:, gold_head_start].shape)
        if self.matcher == "avg":
            cost = - self.cost_relation * pred_rel[:, gold_rel] - self.cost_head * 1 / 2 * (
                    pred_head_start[:, gold_head_start] + pred_head_end[:,
                                                          gold_head_end]) - self.cost_tail * 1 / 2 * (
                           pred_tail_start[:, gold_tail_start] + pred_tail_end[:, gold_tail_end])
        elif self.matcher == "min":
            cost = torch.cat([pred_head_start[:, gold_head_start].unsqueeze(1), pred_rel[:, gold_rel].unsqueeze(1),
                              pred_head_end[:, gold_head_end].unsqueeze(1),
                              pred_tail_start[:, gold_tail_start].unsqueeze(1),
                              pred_tail_end[:, gold_tail_end].unsqueeze(1)], dim=1)
            cost = - torch.min(cost, dim=1)[0]
        else:
            raise ValueError("Wrong matcher")

        # print("1. the shape of cost: ", cost.shape)

        cost = cost.view(bsz, num_generated_triples, -1).cpu()

        # print("2. the shape of cost: ", cost.shape)

        num_gold_triples = [len(v["relation"]) for v in targets]

        result_tuple = []
        for i, c in enumerate(cost.split(num_gold_triples, -1)):

            # Create a new model
            model = Model("AssignmentProblem")
            model.setParam('display/verblevel', 0)

            # Define the decision variables
            n = num_gold_triples[i]
            # print("c[i]:\n", c[i])
            k = len(c[i])
            pred_list = []
            target_list = []

            x = {}
            cost_1 = c[i].detach().numpy()
            # cost_1 = c[i].detach().abs().numpy()
            # print(f"in {i}, cost_1: \n {cost_1}")
            # print("cost_1: \n", cost_1)
            for i in range(n):
                for j in range(k):
                    # vtpe: B means binary
                    x[i, j] = model.addVar(vtype="B", name=f"x_{i}_{j}")

            # print("x: \n", x)
            # Add the constraints
            # i: n_gold_triples, j: num_generated_triples


            for j in range(k):
                model.addCons(quicksum(x[i, j] for i in range(n)) == 1)

            if n > k:
                print("n_class: ", n, " > n_generated_triple: ", k)
                for i in range(n):
                    model.addCons(quicksum(x[i, j] for j in range(k)) <= 1)

            else:
                for i in range(n):
                    model.addCons(quicksum(x[i, j] for j in range(k)) >= 1)

            # Objective function: minimize the total cost
            model.setObjective(quicksum(cost_1[j][i] * x[i, j] for i in range(n) for j in range(k)), "minimize")

            # Solve the problem
            model.optimize()
            status = model.getStatus()
            # for i in range(n):
            #     for j in range(k):
            #         value = model.getVal(x[i, j])
            #         print(f"x[{i}][{j}] = {value}")
            if status != "optimal":
                raise ValueError(f"Optimization did not converge. Status: {status}")

            if status == "optimal":
                for j in range(k):
                    for i in range(n):
                        if model.getVal(x[i, j]) > 0.5:
                            pred_list.append(j)
                            target_list.append(i)

            result_tuple.append(
                (torch.tensor(np.array(pred_list, dtype=np.int64)), torch.tensor(np.array(target_list, dtype=np.int64))))
        # i: row, j: col, if bsz > 1 [(tuple for batch 1), (tuple for batch 2), ...]
        return result_tuple

class ILPMatcher_dotproduct(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, loss_weight, matcher):
        super().__init__()
        self.cost_relation = loss_weight["relation"]
        self.cost_head = loss_weight["head_entity"]
        self.cost_tail = loss_weight["tail_entity"]
        self.matcher = matcher

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_rel_logits": Tensor of dim [batch_size, num_generated_triples, num_classes] with the classification logits
                 "{head, tail}_{start, end}_logits": Tensor of dim [batch_size, num_generated_triples, seq_len] with the predicted index logits
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_generated_triples, num_gold_triples)
        """
        bsz, num_generated_triples = outputs["pred_rel_logits"].shape[:2]
        # We flatten to compute the cost matrices in a batch
        pred_rel = outputs["pred_rel_logits"].flatten(0, 1).softmax(
            -1)  # [bsz * num_generated_triples, num_classes]
        gold_rel = torch.cat([v["relation"] for v in targets])
        # print("pred_rel: \n", pred_rel)
        # print("gold_rel:\n ", gold_rel)
        # after masking the pad token
        # replace all the nan value with -1e9 in the tensor outputs["head_start_logits"]
        outputs["head_start_logits"] = torch.where(torch.isnan(outputs["head_start_logits"]), torch.full_like(outputs["head_start_logits"], -1e9), outputs["head_start_logits"])
        pred_head_start = outputs["head_start_logits"].flatten(0, 1).softmax(-1)  # [bsz * num_generated_triples, seq_len]
        outputs["head_end_logits"] = torch.where(torch.isnan(outputs["head_end_logits"]), torch.full_like(outputs["head_end_logits"], -1e9), outputs["head_end_logits"])
        pred_head_end = outputs["head_end_logits"].flatten(0, 1).softmax(-1)
        outputs["tail_start_logits"] = torch.where(torch.isnan(outputs["tail_start_logits"]), torch.full_like(outputs["tail_start_logits"], -1e9), outputs["tail_start_logits"])
        pred_tail_start = outputs["tail_start_logits"].flatten(0, 1).softmax(-1)
        outputs["tail_end_logits"] = torch.where(torch.isnan(outputs["tail_end_logits"]), torch.full_like(outputs["tail_end_logits"], -1e9), outputs["tail_end_logits"])
        pred_tail_end = outputs["tail_end_logits"].flatten(0, 1).softmax(-1)

        gold_head_start = torch.cat([v["head_start_index"] for v in targets])
        gold_head_end = torch.cat([v["head_end_index"] for v in targets])
        gold_tail_start = torch.cat([v["tail_start_index"] for v in targets])
        gold_tail_end = torch.cat([v["tail_end_index"] for v in targets])
        # print("pred_rel: \n", pred_rel.shape)
        # print("pred_rel[:, gold_rel]: ", pred_rel[:, gold_rel].shape)
        # print("pred_head_start[:, gold_head_start]: ", pred_head_start[:, gold_head_start].shape)
        if self.matcher == "avg":
            cost_rel = get_dot_cost(pred_rel, gold_rel)
            cost_head_start = get_dot_cost(pred_head_start, gold_head_start)
            cost_head_end = get_dot_cost(pred_head_end, gold_head_end)
            cost_tail_start = get_dot_cost(pred_tail_start, gold_tail_start)
            cost_tail_end = get_dot_cost(pred_tail_end, gold_tail_end)

            cost = - self.cost_relation * cost_rel - self.cost_head * 1/2 * (cost_head_start + cost_head_end) - self.cost_tail * 1/2 * (cost_tail_start + cost_tail_end)

        elif self.matcher == "min":
            cost = torch.cat([pred_head_start[:, gold_head_start].unsqueeze(1), pred_rel[:, gold_rel].unsqueeze(1),
                              pred_head_end[:, gold_head_end].unsqueeze(1),
                              pred_tail_start[:, gold_tail_start].unsqueeze(1),
                              pred_tail_end[:, gold_tail_end].unsqueeze(1)], dim=1)
            cost = - torch.min(cost, dim=1)[0]
        else:
            raise ValueError("Wrong matcher")

        # print("1. the shape of cost: ", cost.shape)

        cost = cost.view(bsz, num_generated_triples, -1).cpu()

        # print("2. the shape of cost: ", cost.shape)

        num_gold_triples = [len(v["relation"]) for v in targets]

        result_tuple = []
        for i, c in enumerate(cost.split(num_gold_triples, -1)):

            # Create a new model
            model = Model("AssignmentProblem")
            model.setParam('display/verblevel', 0)

            # Define the decision variables
            n = num_gold_triples[i]
            # print("c[i]:\n", c[i])
            k = len(c[i])
            pred_list = []
            target_list = []

            x = {}
            cost_1 = c[i].detach().numpy()
            # cost_1 = c[i].detach().abs().numpy()
            # print(f"(dot_production) in {i}, cost_1: \n {cost_1}")
            # print("cost_1: \n", cost_1)
            for i in range(n):
                for j in range(k):
                    # vtpe: B means binary
                    x[i, j] = model.addVar(vtype="B", name=f"x_{i}_{j}")

            # print("x: \n", x)
            # Add the constraints
            # i: n_gold_triples, j: num_generated_triples

            for j in range(k):
                model.addCons(quicksum(x[i, j] for i in range(n)) == 1)

            if n > k:
                print("n_class: ", n, " > n_generated_triple: ", k)
                for i in range(n):
                    model.addCons(quicksum(x[i, j] for j in range(k)) <= 1)

            else:
                for i in range(n):
                    model.addCons(quicksum(x[i, j] for j in range(k)) >= 1)

            # Objective function: minimize the total cost
            model.setObjective(quicksum(cost_1[j][i] * x[i, j] for i in range(n) for j in range(k)), "minimize")

            # Solve the problem
            model.optimize()
            status = model.getStatus()
            # for i in range(n):
            #     for j in range(k):
            #         value = model.getVal(x[i, j])
            #         print(f"x[{i}][{j}] = {value}")
            if status != "optimal":
                raise ValueError(f"Optimization did not converge. Status: {status}")

            if status == "optimal":
                for j in range(k):
                    for i in range(n):
                        if model.getVal(x[i, j]) > 0.5:
                            pred_list.append(j)
                            target_list.append(i)

            result_tuple.append(
                (torch.tensor(np.array(pred_list, dtype=np.int64)),
                 torch.tensor(np.array(target_list, dtype=np.int64))))
        # i: row, j: col, if bsz > 1 [(tuple for batch 1), (tuple for batch 2), ...]
        return result_tuple
