"""==================================================================================================="""
################### LIBRARIES ###################
### Basic Libraries
import warnings

warnings.filterwarnings("ignore")

import os, numpy as np, argparse
import time, random
import matplotlib

matplotlib.use('agg')

from tqdm import tqdm

# import configs.par_r50_128d_multisim_fsa as par
import configs.par_r50_512d_agwp as par
### Load Remaining Libraries that need to be loaded after comet_ml
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import models as archs
import datasampler as dsamplers
import datasets as Datasets
import criteria as criteria
import metrics as Metrics
import batchminer as bmine
import evaluations as eval
from utilities import misc
from utilities import logger


def get_args():
    """==================================================================================================="""
    ################### INPUT ARGUMENTS ###################
    # 执行时的命令会统一读入parser中并按阶段解析
    parser = argparse.ArgumentParser()

    parser = par.basic_training_parameters(parser)
    parser = par.batch_creation_parameters(parser)
    parser = par.batchmining_specific_parameters(parser)
    parser = par.loss_specific_parameters(parser)
    # parser = par.wandb_parameters(parser)

    opt = parser.parse_args()
    opt.log_online = False
    """==================================================================================================="""
    ### The following setting is useful when logging to wandb and running multiple seeds per setup:
    ### By setting the savename to <group_plus_seed>, the savename will instead comprise the group and the seed!
    if opt.savename == 'group_plus_seed':
        if opt.log_online:
            opt.savename = opt.group + '_s{}'.format(opt.seed)
        else:
            opt.savename = ''
    return opt


def main():
    opt = get_args()
    # Train start
    full_training_start_time = time.time()

    # Add suffix to dataset path according to used dataset
    opt.source_path += '/' + opt.dataset
    opt.save_path += '/' + opt.dataset

    # Assert that the construction of the batch makes sense, i.e. the division into class-subclusters.
    assert not opt.bs % opt.samples_per_class, 'Batchsize needs to fit number of samples per class for distance sampling and margin/triplet loss!'

    opt.pretrained = not opt.not_pretrained
    opt.pretrained = True
    """==================================================================================================="""
    ################### GPU SETTINGS ###########################
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # if not opt.use_data_parallel:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu[0])
    ### If wandb-logging is turned on, initialize the wandb-run here:

    """==================================================================================================="""
    #################### SEEDS FOR REPROD. #####################
    torch.backends.cudnn.deterministic = True
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    """==================================================================================================="""
    ##################### NETWORK SETUP ##################
    opt.device = torch.device('cuda')
    model = archs.select(opt.arch, opt)



    _ = model.to(opt.device)

    checkpoint = torch.load(opt.checkpoint_path)

    _ = model.load_state_dict(checkpoint["state_dict"])

    """============================================================================"""
    #################### DATALOADER SETUPS ##################
    dataloaders = {}
    datasets = Datasets.select(opt.dataset, opt, opt.source_path)

    dataloaders['evaluation'] = torch.utils.data.DataLoader(datasets['evaluation'], num_workers=opt.kernels,
                                                            batch_size=opt.bs, shuffle=False)
    dataloaders['testing'] = torch.utils.data.DataLoader(datasets['testing'], num_workers=opt.kernels,
                                                         batch_size=opt.bs,
                                                         shuffle=False)
    if opt.use_tv_split:
        dataloaders['validation'] = torch.utils.data.DataLoader(datasets['validation'], num_workers=opt.kernels,
                                                                batch_size=opt.bs, shuffle=False)

    train_data_sampler = dsamplers.select(opt.data_sampler, opt, datasets['training'].image_dict,
                                          datasets['training'].image_list)
    if train_data_sampler.requires_storage:
        train_data_sampler.create_storage(dataloaders['evaluation'], model, opt.device)

    dataloaders['training'] = torch.utils.data.DataLoader(datasets['training'], num_workers=opt.kernels,
                                                          batch_sampler=train_data_sampler)

    opt.n_classes = len(dataloaders['training'].dataset.avail_classes)



    """============================================================================"""
    #################### METRIC COMPUTER ####################
    opt.rho_spectrum_embed_dim = opt.embed_dim
    metric_computer = Metrics.MetricComputer(opt.evaluation_metrics, opt)

    """============================================================================"""
    ################### Summary #########################3
    data_text = 'Dataset:\t {}'.format(opt.dataset.upper())
    setup_text = 'Objective:\t {}'.format(opt.loss.upper())
    arch_text = 'Backbone:\t {} (#weights: {})'.format(opt.arch.upper(), misc.gimme_params(model))

    """============================================================================"""
    ################### SCRIPT MAIN ##########################
    print('\n-----\n')

    iter_count = 0
    loss_args = {'batch': None, 'labels': None, 'batch_features': None, 'f_embed': None}
    sub_loggers = ['Train', 'Test', 'Model Grad']
    LOG = logger.LOGGER(opt, sub_loggers=sub_loggers, start_new=True, log_online=opt.log_online)
    """======================================="""
    ### Evaluate Metric for Training & Test (& Validation)
    model.eval()
    print('\nComputing Testing Metrics...')
    eval.evaluate_only(opt.dataset, LOG, metric_computer, [dataloaders['testing']], model, opt, opt.evaltypes,
                  opt.device, checkpoint,
                  log_key='Test')



if __name__ == '__main__':
    main()
