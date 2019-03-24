import torch.nn.functional as F


def cross_entropy2d(output, target, weight=None):
    n, c, h, w = output.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between output and target
    if h != ht and w != wt:  # upsample labels
        output = F.interpolate(
            output, size=(ht, wt), mode="bilinear", align_corners=True)

    output = output.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(output, target, weight=weight)

    return loss
