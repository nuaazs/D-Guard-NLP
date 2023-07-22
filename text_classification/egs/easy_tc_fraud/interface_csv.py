import gradio as gr
from dguard_nlp.interface.pretrained import load_by_name,inference
from tqdm import tqdm
embedding_model, classifier = load_by_name('bert_cos_b1_entropyloss',device='cuda:0')
def classify_text(text_list):
    results = inference(embedding_model, classifier, text_list, print_result=False)
    # label = "涉诈" if results[0][0] == 1 else "非涉诈"
    # confidence = results[0][1]
    # print(f"分类结果：{label}，置信度：{confidence}")
    return results

# /home/zhaosheng/bert_fraud_classify/text_classification/data/16w_data.csv
# 读取csv文件，遍历其中每一行
# index,label,sentence
# 其中label为1代表涉诈
# 其中label为0代表非涉诈
# 遍历所有行，记录下推理结果与真实标签的不同的结果
import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('--csv_path', type=str, default='/home/zhaosheng/bert_fraud_classify/text_classification/data/16w_data.csv',help='')
parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
args = parser.parse_args()
import logging

# rm csv_process.log if exists
import os
if os.path.exists('csv_process.log'):
    os.remove('csv_process.log')
# init logger, save to csv_process.log
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('csv_process.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)

if __name__ == '__main__':
    with open(args.csv_path,'r') as f:
        lines = f.readlines()
    lines = lines[1:]
    batch_num = len(lines)//args.batchsize
    tn,tp,fn,fp = 0,0,0,0
    pbar = tqdm(total=batch_num)
    # pbar = tqdm(range(batch_num))
    for batch_index in range(batch_num):
        tiny_data = lines[batch_index*args.batchsize:(batch_index+1)*args.batchsize]
        texts = [line.split(',')[2] for line in tiny_data]
        # print(texts)
        labels = [line.split(',')[1] for line in tiny_data]
        predicted = classify_text(texts) # shape (number,2)
        predicted_label = [p[0] for p in predicted]
        assert len(predicted_label) == len(labels)
        for i in range(len(predicted_label)):
            if labels[i] == '1':
                if predicted_label[i] == 1:
                    tp += 1
                else:
                    fn += 1
                    logger.info(f"FN,{texts[i]}")
            else:
                if predicted_label[i] == 1:
                    fp += 1
                    logger.info(f"FP,{texts[i]}")
                else:
                    tn += 1
        pbar.update(1)
        pbar.set_description(f"TP:{tp},TN:{tn},FP:{fp},FN:{fn} Accuracy:{(tp+tn)*100/(tp+tn+fp+fn):.1f}%")
    print(f"TP:{tp},TN:{tn},FP:{fp},FN:{fn} Accuracy:{(tp+tn)*100/(tp+tn+fp+fn):.1f}%")
    # print(f"")
    print(f"Precision:{tp/(tp+fp)}")
    print(f"Recall:{tp/(tp+fn)}")
    # update pbar : TP:{tp},TN:{tn},FP:{fp},FN:{fn}
    # update pbar : Accuracy:{(tp+tn)/(tp+tn+fp+fn)}
    
