import torch
import random
import numpy as np

def init_seed(seed):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def init_device(opt):
    if torch.cuda.is_available():
        opt.cuda = True
        torch.cuda.set_device(int(opt.device[5]))
    else:
        opt.cuda = False
        opt.device = 'cpu'
    return opt

def init_optim(model, opt):
    '''
    Initialize optimizer
    '''
    return torch.optim.Adam(params=model.parameters(),lr=opt.lr_init)

def init_lr_scheduler(optim, opt):
    '''
    Initialize the learning rate scheduler
    '''
    #return torch.optim.lr_scheduler.StepLR(optimizer=optim,gamma=opt.lr_scheduler_rate,step_size=opt.lr_scheduler_step)
    return torch.optim.lr_scheduler.MultiStepLR(optimizer=optim, milestones=opt.lr_decay_steps,
                                                gamma = opt.lr_scheduler_rate)

def print_model_parameters(model, logger, only_num = True):
    logger.info('*****************Model Parameter*****************')
    if not only_num:
        for name, param in model.named_parameters():
            logger.info('{} {} {}'.format(name, param.shape, param.requires_grad))
    total_num = sum([param.nelement() for param in model.parameters()])
    logger.info('Total params num: {}'.format(total_num))
    logger.info('*****************Finish Parameter****************')

def get_memory_usage(device):
    allocated_memory = torch.cuda.memory_allocated(device) / (1024*1024.)
    cached_memory = torch.cuda.memory_cached(device) / (1024*1024.)
    print('Allocated Memory: {:.2f} MB, Cached Memory: {:.2f} MB'.format(allocated_memory, cached_memory))
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    return allocated_memory, cached_memory

def init_parameters(model):
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
        else:
            torch.nn.init.uniform_(p)
    return model

def MAE_torch(output, label):
    return torch.mean(torch.abs(output - label))

def RMSE_torch(output, label):
    return torch.sqrt(torch.mean((output - label)**2))

def MAPE_torch(output, label):
    mask = ~torch.isnan(label)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = 2.0 * (torch.abs(output - label) / (torch.abs(output) + torch.abs(label)))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)
    # return torch.mean(torch.abs(output - label)/label)


def All_Metrics(pred, true, mask1, mask2):
    mae  = MAE_torch(pred, true)
    rmse = RMSE_torch(pred, true)
    mape = MAPE_torch(pred, true)
    rrse = 0.0
    corr = 0.0
    return mae, rmse, mape, rrse, corr

