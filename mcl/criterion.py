import torch
from torch import nn
from .memory import ContrastMemory

eps = 1e-7


class MCL_Loss(nn.Module):
    def __init__(self, args):
        super(MCL_Loss, self).__init__()
        self.embed_list = nn.ModuleList([])
        self.args = args
        for i in range(args.num_branches):
            self.embed_list.append(Embed(args.rep_dim, args.feat_dim))
        self.contrast = ContrastMemory(args.num_branches,
                                       args.feat_dim,
                                       args.n_data,
                                       args.nce_k,
                                       args.nce_t,
                                       args.nce_m)
        self.criterion = ContrastLoss(args.n_data)

    def forward(self, embedings, idx, contrast_idx=None):

        for i in range(self.args.num_branches):
            embedings[i] = self.embed_list[i](embedings[i])
        outs = self.contrast(embedings, idx, contrast_idx)

        loss = 0.
        for out in outs:
            loss = loss + self.criterion(out)
        return loss


class ContrastLoss(nn.Module):
    """
    contrastive loss, corresponding to Eq (11)
    """
    def __init__(self, n_data):
        super(ContrastLoss, self).__init__()
        self.n_data = n_data

    def forward(self, x):
        bsz = x.shape[0]
        m = x.size(1) - 1

        # noise distribution
        Pn = 1 / float(self.n_data)

        # loss for positive pair
        P_pos = x.select(1, 0)
        log_D1 = torch.div(P_pos, P_pos.add(m * Pn + eps)).log_()

        # loss for K negative pair
        P_neg = x.narrow(1, 1, m)
        log_D0 = torch.div(P_neg.clone().fill_(m * Pn), P_neg.add(m * Pn + eps)).log_()

        loss = - (log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / bsz

        return loss


class Embed(nn.Module):
    """Embedding module"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(Embed, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        x = self.l2norm(x)
        return x


class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
