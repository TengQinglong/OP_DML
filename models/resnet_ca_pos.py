"""
The network architectures and weights are adapted and used from the great https://github.com/Cadene/pretrained-backbones.pytorch.
"""
import pretrainedmodels as ptm
import torch
import torch.nn as nn
from .AttentionLayer import CrossAttentionLayer

"""============================================================="""


class Network(torch.nn.Module):
    def __init__(self, opt):
        super(Network, self).__init__()

        self.pars = opt
        self.backbone = ptm.__dict__['resnet50'](num_classes=1000,
                                              pretrained='imagenet' if not opt.not_pretrained else None)
        self.train_flag = True
        self.name = opt.arch

        if 'frozen' in opt.arch:
            for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.backbone.modules()):
                module.eval()
                module.train = lambda _: None

        self.backbone.last_linear = torch.nn.Linear(self.backbone.last_linear.in_features, opt.embed_dim)

        self.layer_blocks = nn.ModuleList([self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4])

        self.feature_tokenizer = nn.Linear(2048, opt.embed_dim)
        self.attention_layers = nn.ModuleList(
            [CrossAttentionLayer(q_dim=opt.embed_dim, k_dim=opt.embed_dim, d_model=opt.d_model, n_head=opt.n_head, residual=True)
             for _ in range(opt.atten_layers)])

        self.out_adjust = None

    def get_pairs(self, batch, labels):
        batch_size = len(labels)
        batch_pos = []
        batch_neg = []
        for i in range(batch_size):
            pos_idxs = labels == labels[i]
            pos_idxs[i] = 0
            neg_idxs = labels != labels[i]

            batch_pos.append(pos_idxs)
            batch_neg.append(neg_idxs)
        return batch_pos, batch_neg

    def forward(self, x, **kwargs):
        x = self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x))))
        for layerblock in self.layer_blocks:
            x = layerblock(x)
        no_avg_feat = x
        x = self.backbone.avgpool(x)
        enc_out = x = x.view(x.size(0), -1)

        x = self.backbone.last_linear(x)
        if self.train_flag:
            labels = kwargs["labels"]
            pos_pair, _ = self.get_pairs(x, labels)
            query = x.unsqueeze(1)
            no_avg_feat_tokens = self.feature_tokenizer(no_avg_feat.view(no_avg_feat.size(0), no_avg_feat.size(1), -1).permute(0, 2, 1))
            key = []
            for i in range(len(pos_pair)):
                key.append(no_avg_feat_tokens[pos_pair[i]][0])
            key = torch.stack(key)
            # atten_input = torch.cat(x.unsqueeze(-1), no_avg_feat_tokens)
            for i, layer in enumerate(self.attention_layers):
                query = layer(query, key, key)
            x = query.squeeze(1)
        if 'normalize' in self.pars.arch:
            x = torch.nn.functional.normalize(x, dim=-1)
        # if self.out_adjust and not self.train:
        #     x = self.out_adjust(x)

        return x, (enc_out, no_avg_feat)
