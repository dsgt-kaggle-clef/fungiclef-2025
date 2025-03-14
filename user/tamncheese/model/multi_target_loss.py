import torch
import torch.nn.functional as F

POISON_FALSE_NEGATIVE_WEIGHT = 10


def loss(self, pred, target, smoothing=0.1):
    preds = torch.hsplit(pred, self.idx_splits)
    targets = torch.split(target, 1, dim=1)
    losses = []
    for i, (p, t) in enumerate(zip(preds, targets)):
        t = t.squeeze(1).long()  # Ensure correct shape
        weight = (
            self.class_weights[i].float() if self.class_weights[i] is not None else None
        )
        if weight is not None:
            weight = weight.to(p.device)
        if p.shape[-1] == 1:
            p = p.squeeze(1)
            t = t.float()
            pos_weight = self.dynamic_pos_weight(
                t, scale=POISON_FALSE_NEGATIVE_WEIGHT
            ).to(p.device)
            loss = F.binary_cross_entropy_with_logits(p, t, pos_weight=pos_weight)
        else:
            if smoothing > 0:
                # loss = label_smoothing_loss(p, t, smoothing)
                pass
            else:
                loss = F.cross_entropy(p, t, weight=weight)
        losses.append(loss * self.output_weights[i])
    return sum(losses)
