import models.resnet50
import models.googlenet
import models.bninception
import models.resnet_agwp
import models.resnet_fsa


def select(arch, opt):
    if "resnet_fsa" in arch:
        return resnet_fsa.Network(opt)
    if "resnet_agwp" in arch:
        return resnet_agwp.Network(opt)
    if 'resnet50' in arch:
        return resnet50.Network(opt)
    if 'googlenet' in arch:
        return googlenet.Network(opt)
    if 'bninception' in arch:
        return bninception.Network(opt)
