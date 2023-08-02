import torch
import yaml
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from dguard_nlp.utils import load_yaml2dict
from dguard_nlp.utils.seed import set_seed
from dguard_nlp.models.bert import BertClassificationModel
from dguard_nlp.models.classify import CosineClassifier
from dguard_nlp.optimizer.optimizers import get_optimizer
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
args = parser.parse_args()

if __name__ == '__main__':
    # args get config.yaml path
    args = parser.parse_args()
    config = load_yaml2dict(args.config)
    set_seed(config['seed'])
    if config['device'] != "cpu":
        torch.cuda.set_device(config['device'])
    train_loader, vali_loader, test_loader = get_dataframe(config)
    exp_name = config['exp']
    exp_path = f'./exp/{exp_name}'
    os.makedirs(exp_path, exist_ok=True)
    # save config to exp_path
    with open(f'{exp_path}/config.yaml', 'w') as f:
        yaml.dump(config, f)

    # add logger, save to exp_path/train.log and print to console
    logger = get_logger(f'{exp_path}/train.log')
    logger.info(f"Load data done! Train: {len(train_loader)} Vali: {len(vali_loader)} Test: {len(test_loader)}")

    model = BertClassificationModel(device=config['device'])
    cos_model = CosineClassifier(input_dim=768,num_blocks=3,inter_dim=512,out_neurons=2).to(config['device'])
    pipeline = torch.nn.Sequential(model,cos_model)
    
    logger.info(f"Model has {sum(p.numel() for p in pipeline.parameters()):,} parameters")
    logger.info(f"Model has {sum(p.numel() for p in pipeline.parameters() if p.requires_grad):,} trainable parameters")
    logger.info(f"Device: {config['device']}")
    # optimizer = #get_optimizer(config, pipeline)
    
    config_class = Config(config)
    optimizer = build('optimizer', config_class)
    config_class.lr_scheduler['args']['step_per_epoch'] = len(train_loader)
    # print(config["lr_scheduler"])
    lr_scheduler = build('lr_scheduler', config_class)

    model, optimizer = train(config, pipeline, optimizer, lr_scheduler,train_loader, vali_loader,logger=logger,device=config['device'])
    r = test(pipeline, test_loader)
    print(f">>> The accuracy of the model on the test set is: {r * 100:.2f}%")
