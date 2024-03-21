"""
The network architectures and weights are adapted and used from the great https://github.com/Cadene/pretrained-backbones.pytorch.
"""
import pretrainedmodels as ptm
import torch
import torch.nn as nn
from .AttentionLayer import CrossAttentionLayer, CrossAttentionFFNLayer

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
            [CrossAttentionFFNLayer(q_dim=opt.embed_dim, k_dim=opt.embed_dim, d_model=opt.d_model, n_head=opt.n_head, residual=True)
             for _ in range(opt.atten_layers)])

        self.out_adjust = None

    def forward(self, x, **kwargs):
        x = self.backbone.maxpool(self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x))))
        for layerblock in self.layer_blocks:
            x = layerblock(x)
        no_avg_feat = x
        x = self.backbone.avgpool(x)
        enc_out = x = x.view(x.size(0), -1)

        x = self.backbone.last_linear(x)
        if self.train_flag:
            results = []
            query = x.unsqueeze(1)
            no_avg_feat_tokens = self.feature_tokenizer(no_avg_feat.view(no_avg_feat.size(0), no_avg_feat.size(1), -1).permute(0, 2, 1))
            for i, key in enumerate(no_avg_feat_tokens):  # key:(hw, embed_dim)
                cur_query = query  # bs,1,embed_dim
                key = key.unsqueeze(0).repeat(cur_query.shape[0], 1, 1)
                for j, layer in enumerate(self.attention_layers):
                    cur_query = layer(cur_query, key, key)  # bs,1,embed_dim
                if 'normalize' in self.pars.arch:
                    cur_query = torch.nn.functional.normalize(cur_query, dim=-1)
                results.append(cur_query)
            return torch.cat(results, dim=1)


        if 'normalize' in self.pars.arch:
            x = torch.nn.functional.normalize(x, dim=-1)
        # if self.out_adjust and not self.train:
        #     x = self.out_adjust(x)

        return x, (enc_out, no_avg_feat)
