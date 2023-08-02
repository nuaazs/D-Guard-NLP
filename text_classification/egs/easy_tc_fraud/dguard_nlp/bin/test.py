from tqdm import tqdm
import torch
from torch import nn
import dguard_nlp
@dguard_nlp.utils.timeit
def test(model,test_loader):
    model.eval()
    correct = 0
    total = 0
    for i,(data,labels) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            out=model(data) # [batch_size,num_class]

        out = out.argmax(dim=1)
        correct += (out.cpu() == labels).sum().item()
        total += len(labels)
    print(f">>> The accuracy of the model on the test set is: {correct / total * 100:.2f}%")
    return correct / total * 100