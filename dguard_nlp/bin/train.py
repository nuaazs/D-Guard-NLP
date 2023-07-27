from tqdm import tqdm
import torch
from torch import nn
import dguard_nlp
loss_func=torch.nn.CrossEntropyLoss()

@dguard_nlp.utils.timeit
def train(epochs,valid_interval,model,optimizer,train_loader,vali_loader):
    for epoch in range(epochs):
        pbar = tqdm(train_loader)
        pbar.set_description(f"Epoch {epoch}")
        for i,(data,labels) in enumerate(pbar):
            model.train()
            out=model(data) # [batch_size,num_class]
            loss=loss_func(out.cpu(),labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i%5==0:
                out=out.argmax(dim=-1)
                acc=(out.cpu()==labels).sum().item()/len(labels)
                # set tqdm prefix
                pbar.set_postfix(Loss=loss.item(),ACC=acc, refresh=False)        
        if epoch%valid_interval==0:
            # start valid
            model.eval()
            correct = 0
            total = 0
            for i,(data,labels) in enumerate(vali_loader):
                with torch.no_grad():
                    out=model(data)
                out = out.argmax(dim=1)
                correct += (out.cpu() == labels).sum().item()
                total += len(labels)
            acc_valid = correct / total
            print(f">>> Epoch {epoch} - Val Loss: {loss.item()} - Val ACC: {acc_valid}")

    return model,optimizer