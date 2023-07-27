import torch
from sklearn.model_selection import train_test_split
from dguard_nlp.utils.seed import set_seed
from transformers import AdamW
set_seed(100)
from dguard_nlp.models.bert import BertClassificationModel
from dguard_nlp.optimizer.optimizers import get_optimizer
from dguard_nlp.datasets.preprocess import get_dataframe
from dguard_nlp.bin.train import train
from dguard_nlp.bin.test import test


# device
device = "cuda:1" # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# args
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--batchsize', type=int, default=1,help='')
parser.add_argument('--valid_interval', type=int, default=1,help='')
parser.add_argument('--epochs', type=int, default=100,help='')
parser.add_argument('--train_csv', type=str, default="../../data/weibo_senti_100k_sentence.csv",help='')
parser.add_argument('--test_csv', type=str, default=None,help='')
parser.add_argument('--lr', type=float, default=1e-4,help='')

args = parser.parse_args()

if __name__ == '__main__':
    train_loader,vali_loader,test_loader=get_dataframe(train_csv=args.train_csv,test_csv=args.test_csv,batch_size=args.batchsize,check_data=False)
    print(f"Load data done! Train: {len(train_loader)} Vali: {len(vali_loader)} Test: {len(test_loader)}")
    model=BertClassificationModel()
    model=model.to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    print(f"Device: {device}")
    optimizer=get_optimizer("AdamW",model,args.lr)    
    model,optimizer = train(args.epochs,args.valid_interval,model,optimizer,train_loader,vali_loader)
    r = test(model,test_loader)
