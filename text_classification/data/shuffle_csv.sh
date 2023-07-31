#!/bin/bash

# 读取文件内容，并将第一行保存到变量header中
header=$(head -n 1 16w_data.csv)

# 打乱文件内容（除第一行外）
tail -n +2 16w_data.csv | shuf > temp.csv

# 将打乱后的内容与第一行合并为新的文件output.txt
echo "$header" > 16w_formated.csv
cat temp.csv >> 16w_formated.csv

# 去除文件中的中文标点符号
sed -i 's/[，。？！【】《》（）：；]//g' 16w_formated.csv

# 删除临时文件
rm temp.csv

