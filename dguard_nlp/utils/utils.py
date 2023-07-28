import time
import gzip
from Crypto.Cipher import AES
import os
import subprocess
def timeit(func):
    def wrapper(*args,**kwargs):
        szipt=time.time()
        func(*args,**kwargs)
        print(f">>> Time used: {time.time()-szipt:.2f}s")
    return wrapper


import yaml
iv = b'0000000000000000'

def load_yaml2dict(yaml_file):
    with open(yaml_file) as f:
        yaml_dict = yaml.safe_load(f)
    return yaml_dict


def encrypt_compress_file(input_file, output_file, password):
    with open(input_file, 'rb') as f:
        input_data = f.read()
    cipher = AES.new(password, AES.MODE_CBC, iv)
    padding_size = AES.block_size - len(input_data) % AES.block_size
    input_data += bytes([padding_size]) * padding_size
    encrypted_data = cipher.encrypt(input_data)
    with open(output_file, 'wb') as f:
        f.write(encrypted_data)
    print("encry success")

def decrypt_decompress_file(input_file, output_file, password):
    with open(input_file, 'rb') as f:
        encrypted_data = f.read()
    cipher = AES.new(password, AES.MODE_CBC, iv)
    decrypted_data = cipher.decrypt(encrypted_data)
    padding_size = decrypted_data[-1]
    decrypted_data = decrypted_data[:-padding_size]
    with open(output_file, 'w') as f:
        f.write(decrypted_data.decode())
    print("decry success")