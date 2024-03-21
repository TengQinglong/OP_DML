import torch
import torch.nn as nn
import torch.nn.functional as F

ALLOWED_MINING_OPS  = None
REQUIRES_BATCHMINER = False
REQUIRES_OPTIM      = True

class Criterion(torch.nn.Module):
    def __init__(self, opt):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(opt.n_classes, opt.embed_dim))
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = opt.n_classes
        self.sz_embed = opt.embed_dim
        self.delta = 0.1
        self.alpha = 32

        self.name = 'proxyanchor'

        self.optim_dict_list = [{'params':self.proxies, 'lr':opt.lr * opt.loss_proxynca_lrmulti}]

        self.ALLOWED_MINING_OPS = ALLOWED_MINING_OPS
        self.REQUIRES_BATCHMINER = REQUIRES_BATCHMINER
        self.REQUIRES_OPTIM = REQUIRES_OPTIM

    def norm(self, x, axis=-1):
        x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
        return x
    def forward(self, batch, labels, **kwargs):
        P = self.proxies
        labels.to("cuda")

        # 计算余弦相似度
        labels = labels.unsqueeze(1)

        cos = 1 - torch.mm(self.norm(batch), self.norm(P).permute(1, 0))

        # 生成one-hot标签
        labels_tmp = torch.FloatTensor(labels.shape[0], self.nb_classes).zero_()
        P_one_hot = labels_tmp.scatter(1, labels.data, 1).to("cuda")
        N_one_hot = 1 - P_one_hot

        # 统计有效proxy数量
        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(
            dim=1)  # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies

        # 计算损失函数
        pos_exp = torch.exp(self.alpha * (cos + self.delta))
        neg_exp = torch.exp(-self.alpha * (cos - self.delta))

        P_sim_sum = torch.mul(P_one_hot, pos_exp).sum(dim=0)
        N_sim_sum = torch.mul(N_one_hot, neg_exp).sum(dim=0)

        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term

        return loss

