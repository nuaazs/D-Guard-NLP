from dguard_nlp.utils.utils import encrypt_compress_file
from dguard_nlp.utils.utils import decrypt_decompress_file
import os
filepath = '/home/zhaosheng/bert_fraud_classify/text_classification/data/weibo_senti_100k_sentence.csv'
encrypt_compress_file(filepath, 'output.bin', b'passwordpassword')
decrypt_decompress_file('output.bin', 'output.txt', b'passwordpassword')
os.system('rm output.bin')
