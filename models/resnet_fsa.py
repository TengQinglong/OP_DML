"""
The network architectures and weights are adapted and used from the great https://github.com/Cadene/pretrained-models.pytorch.
"""
import pretrainedmodels as ptm
import torch
import torch.nn as nn
from .AttentionLayer import SelfAttentionLayer
from .transformer.embedding.positional_encoding import PositionalEncoding

"""============================================================="""
class Network(torch.nn.Module):
    def __init__(self, opt):
        super(Network, self).__init__()

        self.pars  = opt
        self.model = ptm.__dict__['resnet50'](num_classes=1000, pretrained='imagenet' if not opt.not_pretrained else None)

        self.name = opt.arch
        self.train_flag = False

        if 'frozen' in opt.arch:
            for module in filter(lambda m: type(m) == nn.BatchNorm2d, self.model.modules()):
                module.eval()
                module.train = lambda _: None

        self.model.last_linear = torch.nn.Linear(self.model.last_linear.in_features, opt.embed_dim)

        self.layer_blocks = nn.ModuleList([self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4])

        self.feature_tokenizer = nn.Linear(2048, opt.embed_dim)
        self.attention_layers = nn.Sequential()
        for i in range(opt.atten_layers):
            self.attention_layers.append(SelfAttentionLayer(input_dim=opt.embed_dim, d_model=opt.d_model, n_head=opt.n_head))
        self.position_embedder = PositionalEncoding(opt.embed_dim, 1000, torch.device("cuda"))

        self.out_adjust = None


    def forward(self, x, **kwargs):
        x = self.model.maxpool(self.model.relu(self.model.bn1(self.model.conv1(x))))
        for layerblock in self.layer_blocks:
            x = layerblock(x)
        no_avg_feat = x
        x = self.model.avgpool(x)
        enc_out = x = x.view(x.size(0),-1)

        x = self.model.last_linear(x)
        if "train_flag" in kwargs.keys():
            self.train_flag = kwargs["train_flag"]
        if self.train_flag:
            results = []
            query = x.unsqueeze(1)
            local_feat_tokens = self.feature_tokenizer(no_avg_feat.view(no_avg_feat.size(0), no_avg_feat.size(1), -1).permute(0, 2, 1))
            position_embedding = self.position_embedder(local_feat_tokens)
            pe_local_feat_tokens = local_feat_tokens + position_embedding

            for i, key in enumerate(pe_local_feat_tokens):  # key:(hw, embed_dim)
                cur_query = query  # bs,1,embed_dim
                key = key.unsqueeze(0).repeat(cur_query.shape[0], 1, 1)
                atten_input = torch.cat((cur_query, key), dim=1)
                for atten_layer in self.attention_layers:
                    atten_input = atten_layer(atten_input)  # (bs, hw+1, embed_dim)
                results.append(atten_input[:, 0, :])
            self.train_flag = False
            return torch.stack(results, dim=0)
        else:
            if 'normalize' in self.pars.arch:
                x = torch.nn.functional.normalize(x, dim=-1)
            if self.out_adjust and not self.train:
                x = self.out_adjust(x)
            return x, (enc_out, no_avg_feat)

