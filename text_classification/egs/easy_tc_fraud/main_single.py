import torch
import yaml
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from dguard_nlp.utils import load_yaml2dict
from dguard_nlp.utils.seed import set_seed
from dguard_nlp.datasets.preprocess import get_dataframe
from dguard_nlp.bin.train import train
from dguard_nlp.bin.test import test
from dguard_nlp.utils import get_logger
from dguard_nlp.utils.config import build_config
from dguard_nlp.utils.builder import build
from dguard_nlp.utils.config import Config
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--config', type=str, default='conf/config.yaml',help='')
parser.add_argument('--resume', default=True, type=bool, help='Resume from recent checkpoint or not')
args = parser.parse_args()

if __name__ == '__main__':

    config = load_yaml2dict(args.config)
    config_class = Config(config)
    set_seed(config_class.seed)

    if config_class.device != "cpu":
        torch.cuda.set_device(config_class.device)
    
    train_loader, vali_loader, test_loader = get_dataframe(config)

    exp_path = f"./exp/{config_class.exp}"
    os.makedirs(exp_path, exist_ok=True)
    logger = get_logger(f'{exp_path}/train.log')
    with open(f'{exp_path}/config.yaml', 'w') as f:
        yaml.dump(config, f)

    logger.info(f"Load data done! Train: {len(train_loader)} Vali: {len(vali_loader)} Test: {len(test_loader)}")
    classifier = build('classifier', config_class)
    embedding_model = build('embedding_model', config_class)
    pipeline = torch.nn.Sequential(embedding_model,classifier)
    
    logger.info(f"Model has {sum(p.numel() for p in pipeline.parameters()):,} parameters")
    logger.info(f"Model has {sum(p.numel() for p in pipeline.parameters() if p.requires_grad):,} trainable parameters")
    logger.info(f"Device: {config_class.device}")
    
    config_class.optimizer['args']['params'] = pipeline.parameters()
    config_class.lr_scheduler['args']['step_per_epoch'] = len(train_loader)

    optimizer = build('optimizer', config_class)
    checkpointer = build('checkpointer', config_class)
    lr_scheduler = build('lr_scheduler', config_class)

    criterion = build('loss', config_class)
    print(criterion)

    if args.resume:
        checkpointer.recover_if_possible(device='cuda')

    model, optimizer = train(config_class, pipeline, optimizer,criterion, lr_scheduler,train_loader, vali_loader,checkpointer,logger=logger)
    r = test(pipeline, test_loader)

    print(f">>> The accuracy of the model on the test set is: {r * 100:.2f}%")
