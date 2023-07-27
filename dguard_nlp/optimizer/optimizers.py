
from transformers import AdamW

def get_optimizer(name,model,lr=1e-4):
    if name == "AdamW":
        return AdamW(model.parameters(),lr=lr)