import torch
import yaml
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from dguard_nlp.utils import load_yaml2dict
from dguard_nlp.utils.seed import set_seed
from dguard_nlp.models.bert import BertClassificationModel
from dguard_nlp.optimizer.optimizers import get_optimizer
from dguard_nlp.datasets.preprocess import get_dataframe
from dguard_nlp.bin.train import train
from dguard_nlp.bin.test import test

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--config', type=str, default='conf/config.yaml',help='')
parser.add_argument('--gpu', nargs='+', default='0 1 2 3 4 5 6 7',help='')
args = parser.parse_args()
rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
gpu = int(args.gpu[rank])
torch.cuda.set_device(gpu)
dist.init_process_group(backend='nccl')
set_seed(100)

if __name__ == '__main__':
    # args get config.yaml path
    args = parser.parse_args()
    config = load_yaml2dict(args.config)

    train_loader, vali_loader, test_loader = get_dataframe(config)
    print(f"Load data done! Train: {len(train_loader)} Vali: {len(vali_loader)} Test: {len(test_loader)}")

    model = BertClassificationModel().to(gpu)
    if torch.cuda.device_count() > 1:
        model = DDP(model)  # Enable multi-GPU training if available

    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    print(f"Device: {gpu}")

    optimizer = get_optimizer(config, model)

    model, optimizer = train(config['epochs'], config['valid_interval'], model, optimizer, train_loader, vali_loader)
    dist.barrier()
    if rank == 0:
        r = test(model, test_loader)
        print(f">>> The accuracy of the model on the test set is: {r * 100:.2f}%")