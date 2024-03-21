"""
The network architectures and weights are adapted and used from the great https://github.com/Cadene/pretrained-models.pytorch.
"""
import pretrainedmodels as ptm
import torch
import torch.nn as nn
from .AttentionLayer import SelfAttentionLayer

"""============================================================="""
class Network(torch.nn.Module):
    def __init__(self, opt):
        super(Network, self).__init__()

        self.pars  = opt
        self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained='imagenet' if not opt.not_pretrained else None)

        self.name = opt.arch

        if 'frozen' in opt.arch:
            for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
                module.eval()
                module.train = lambda _: None

        self.model.last_linear = torch.nn.Linear(self.model.last_linear.in_features, opt.embed_dim)

        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])

        self.feature_tokenizer = nn.Linear(2048, opt.embed_dim)
        self.attention_layers = nn.Sequential([SelfAttentionLayer(d_model=opt.d_model, n_head=opt.n_head, residual=False)
                                               for _ in range(opt.atten_layers)])

        self.out_adjust = None


    def forward(self, x, **kwargs):
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))
        for layerblock in self.layer_blocks:
            x = layerblock(x)
        no_avg_feat = x
        x = self.model.avgpool(x)
        enc_out = x = x.view(x.size(0),-1)

        x = self.model.last_linear(x)
        no_avg_feat_tokens = self.feature_tokenizer(no_avg_feat.view(no_avg_feat.size(0), no_avg_feat.size(1), -1))
        atten_input = torch.cat(x.unsqueeze(-1), no_avg_feat_tokens)
        atten_output = self.attention_layers(atten_input)
        x = atten_output[..., 0]
        if 'normalize' in self.pars.arch:
            x = torch.nn.functional.normalize(x, dim=-1)
        if self.out_adjust and not self.train:
            x = self.out_adjust(x)

        return x, (enc_out, no_avg_feat)

