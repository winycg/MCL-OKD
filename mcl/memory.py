import torch
from torch import nn
import math


class ContrastMemory(nn.Module):
    """
    memory buffer that supplies large amount of negative samples.
    """
    def __init__(self, num_branches, inputSize, outputSize, K, T=0.07, momentum=0.5):
        super(ContrastMemory, self).__init__()
        self.num_branches = num_branches
        self.nLem = outputSize
        self.unigrams = torch.ones(self.nLem)
        self.multinomial = AliasMethod(self.unigrams)
        self.multinomial.cuda()
        self.K = K

        self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        for i in range(num_branches):
            self.register_buffer('memory_v'+str(i), torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, embedings, y, idx=None):
        K = int(self.params[0].item())
        T = self.params[1].item()

        momentum = self.params[4].item()
        batchSize = embedings[0].size(0)
        outputSize = self.memory_v0.size(0)
        inputSize = self.memory_v0.size(1)

        # original score computation
        if idx is None:
            idx = self.multinomial.draw(batchSize * (self.K + 1)).view(batchSize, -1)
            idx.select(1, 0).copy_(y.data)

        out_v = []
        for i in range(self.num_branches):
            for j in range(i+1, self.num_branches):
                weight_v1 = torch.index_select(getattr(self, 'memory_v'+str(i)), 0, idx.view(-1)).detach()
                weight_v1 = weight_v1.view(batchSize, K + 1, inputSize)
                out_v2 = torch.bmm(weight_v1, embedings[j].view(batchSize, inputSize, 1))
                out_v2 = torch.exp(torch.div(out_v2, T))
                out_v.append(out_v2)

                weight_v2 = torch.index_select(getattr(self, 'memory_v'+str(j)), 0, idx.view(-1)).detach()
                weight_v2 = weight_v2.view(batchSize, K + 1, inputSize)
                out_v1 = torch.bmm(weight_v2, embedings[i].view(batchSize, inputSize, 1))
                out_v1 = torch.exp(torch.div(out_v1, T))
                out_v.append(out_v1)

        '''
        # sample
        weight_v1 = torch.index_select(self.memory_v1, 0, idx.view(-1)).detach()
        weight_v1 = weight_v1.view(batchSize, K + 1, inputSize)
        out_v2 = torch.bmm(weight_v1, v2.view(batchSize, inputSize, 1))
        out_v2 = torch.exp(torch.div(out_v2, T))
        # sample
        weight_v2 = torch.index_select(self.memory_v2, 0, idx.view(-1)).detach()
        weight_v2 = weight_v2.view(batchSize, K + 1, inputSize)
        out_v1 = torch.bmm(weight_v2, v1.view(batchSize, inputSize, 1))
        out_v1 = torch.exp(torch.div(out_v1, T))
        '''

        for i in range(len(out_v)):
            z_v = out_v[i].mean().clone().detach().item() * outputSize
            out_v[i] = torch.div(out_v[i], z_v).contiguous()

        '''
        # set Z if haven't been set yet
        if Z_v1 < 0:
            self.params[2] = out_v1.mean() * outputSize
            Z_v1 = self.params[2].clone().detach().item()
            print("normalization constant Z_v1 is set to {:.1f}".format(Z_v1))
        if Z_v2 < 0:
            self.params[3] = out_v2.mean() * outputSize
            Z_v2 = self.params[3].clone().detach().item()
            print("normalization constant Z_v2 is set to {:.1f}".format(Z_v2))
        

        # compute out_v1, out_v2
        out_v1 = torch.div(out_v1, Z_v1).contiguous()
        out_v2 = torch.div(out_v2, Z_v2).contiguous()
        '''

        # update memory
        with torch.no_grad():
            for i in range(len(embedings)):
                pos = torch.index_select(getattr(self, 'memory_v'+str(i)), 0, y.view(-1))
                pos.mul_(momentum)
                pos.add_(torch.mul(embedings[i], 1 - momentum))
                l_norm = pos.pow(2).sum(1, keepdim=True).pow(0.5)
                updated_v = pos.div(l_norm)
                getattr(self, 'memory_v'+str(i)).index_copy_(0, y, updated_v)

        return out_v


class AliasMethod(object):
    def __init__(self, probs):

        if probs.sum() > 1:
            probs.div_(probs.sum())
        K = len(probs)
        self.prob = torch.zeros(K)
        self.alias = torch.LongTensor([0]*K)

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.prob[kk] = K*prob
            if self.prob[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.alias[small] = large
            self.prob[large] = (self.prob[large] - 1.0) + self.prob[small]

            if self.prob[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        for last_one in smaller+larger:
            self.prob[last_one] = 1

    def cuda(self):
        self.prob = self.prob.cuda()
        self.alias = self.alias.cuda()

    def draw(self, N):
        """ Draw N samples from multinomial """
        K = self.alias.size(0)

        kk = torch.zeros(N, dtype=torch.long, device=self.prob.device).random_(0, K)
        prob = self.prob.index_select(0, kk)
        alias = self.alias.index_select(0, kk)
        # b is whether a random number is greater than q
        b = torch.bernoulli(prob)
        oq = kk.mul(b.long())
        oj = alias.mul((1-b).long())

        return oq + oj