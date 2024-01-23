import torch.nn.functional as F
import torch.nn as nn
import torch, math
from models.matcher import HungarianMatcher, ILPMatcher, HungarianMatcher_dotproduct, ILPMatcher_dotproduct


class SetCriterion(nn.Module):
    """ This class computes the loss for Set_RE.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class, subject position and object position)
    """
    def __init__(self, num_classes, loss_weight, na_coef, losses, matcher, use_ILP=False, use_dotproduct=False, none_class=True):
        """ Create the criterion.
        Parameters:
            num_classes: number of relation categories
            matcher: module able to compute a matching between targets and proposals
            loss_weight: dict containing as key the names of the losses and as values their relative weight.
            na_coef: list containg the relative classification weight applied to the NA category and positional classification weight applied to the [SEP]
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.loss_weight = loss_weight
        if use_ILP and use_dotproduct:
            print("using ILPMatcher_dotproduct")
            self.matcher = ILPMatcher_dotproduct(loss_weight, matcher)
        elif use_ILP and not use_dotproduct:
            print("using ILPMatcher")
            self.matcher = ILPMatcher(loss_weight, matcher)
        elif not use_ILP and use_dotproduct:
            print("using HungarianMatcher_dotproduct")
            self.matcher = HungarianMatcher_dotproduct(loss_weight, matcher)
        else:
            print("using HungarianMatcher")
            self.matcher = HungarianMatcher(loss_weight, matcher)

        self.losses = losses
        if use_ILP:
            rel_weight = torch.ones(self.num_classes)
        else:
            if none_class:
                rel_weight = torch.ones(self.num_classes + 1)
                rel_weight[-1] = na_coef
            else:
                rel_weight = torch.ones(self.num_classes)
                print("no none class, coef no meaning.")
        # print("the rel_weight: ", rel_weight)
        self.register_buffer('rel_weight', rel_weight)
        self.use_ILP = use_ILP
        self.none_class = none_class

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets)
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            # print(f"before loss: {loss}")
            # print(f"before losses: {losses}")
            # print(f"outputs: {outputs}")
            # print(f"targets: {targets}")
            # print(f"indices: {indices}")
            if loss == "entity" and self.empty_targets(targets):
                # print("empty targets", targets)
                pass
            else:
                losses.update(self.get_loss(loss, outputs, targets, indices))
            # print(f"after loss: {loss}")
            # print(f"after lossos: {losses}")

        # print("self.loss_weight: ", self.loss_weight)
        losses = sum(losses[k] * self.loss_weight[k] for k in losses.keys() if k in self.loss_weight)
        # print(f"losses: {losses})")
        return losses

    def relation_loss(self, outputs, targets, indices):
        """Classification loss (NLL)
        targets dicts must contain the key "relation" containing a tensor of dim [bsz]
        """
        src_logits = outputs['pred_rel_logits'] # [bsz, num_generated_triples, num_rel+1]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["relation"][i] for t, (_, i) in zip(targets, indices)])

        if self.none_class:
            target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                        dtype=torch.int64, device=src_logits.device)
        else:
            target_classes = torch.full(src_logits.shape[:2], self.num_classes - 1,
                                        dtype=torch.int64, device=src_logits.device)

        target_classes[idx] = target_classes_o
        loss = F.cross_entropy(src_logits.flatten(0, 1), target_classes.flatten(0, 1), weight=self.rel_weight)
        losses = {'relation': loss}
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices):
        """ Basically, this won't be used in the training process.
        Compute the cardinality error, ie the absolute error in the number of predicted non-empty triples
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_rel_logits = outputs['pred_rel_logits']
        device = pred_rel_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        # print(f"pred_rel_logits.argmax(-1): {pred_rel_logits.argmax(-1)}")
        # print(f"pred_rel_logits.shape[-1] - 1: {pred_rel_logits.shape[-1] - 1}")

        card_pred = (pred_rel_logits.argmax(-1) != pred_rel_logits.shape[-1] - 1).sum(1)
        # print(f"card_pred: {card_pred}")
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices,  **kwargs):
        loss_map = {
            'relation': self.relation_loss,
            'cardinality': self.loss_cardinality,
            'entity': self.entity_loss
        }
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def entity_loss(self, outputs, targets, indices):
        """Compute the losses related to the position of head entity or tail entity
        """
        # print(f" We are in the entity_loss() function, the indices:\n {indices}")
        idx = self._get_src_permutation_idx(indices)
        # print(f"We are in the entity_loss() function, after permutation, the idx:\n {idx}")
        selected_pred_head_start = outputs["head_start_logits"][idx]
        # print(f"the shape of outputs['head_start_logits']: {outputs['head_start_logits'].shape}")
        # print(f"the outputs['head_start_logits']: \n{outputs['head_start_logits']}")
        # print("=====================================")
        # print(f"the shape of selected_pred_head_start: {selected_pred_head_start.shape}")
        # print(f"the selected_pred_head_start: \n{selected_pred_head_start}")
        selected_pred_head_end = outputs["head_end_logits"][idx]
        selected_pred_tail_start = outputs["tail_start_logits"][idx]
        selected_pred_tail_end = outputs["tail_end_logits"][idx]

        target_head_start = torch.cat([t["head_start_index"][i] for t, (_, i) in zip(targets, indices)])
        # print(f"zip(targets, indices): \n{zip(targets, indices)}")
        # print(f"target_head_start: \n{target_head_start}")
        target_head_end = torch.cat([t["head_end_index"][i] for t, (_, i) in zip(targets, indices)])
        target_tail_start = torch.cat([t["tail_start_index"][i] for t, (_, i) in zip(targets, indices)])
        target_tail_end = torch.cat([t["tail_end_index"][i] for t, (_, i) in zip(targets, indices)])


        head_start_loss = F.cross_entropy(selected_pred_head_start, target_head_start)
        head_end_loss = F.cross_entropy(selected_pred_head_end, target_head_end)
        tail_start_loss = F.cross_entropy(selected_pred_tail_start, target_tail_start)
        tail_end_loss = F.cross_entropy(selected_pred_tail_end, target_tail_end)
        losses = {'head_entity': 1/2*(head_start_loss + head_end_loss), "tail_entity": 1/2*(tail_start_loss + tail_end_loss)}
        # print(losses)
        return losses

    @staticmethod
    def empty_targets(targets):
        flag = True
        for target in targets:
            if len(target["relation"]) != 0:
                flag = False
                break
        return flag
