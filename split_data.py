import os
import random

file_path="./data1"
train_rate=0.8
def split_data(file_path,train_rate):
    if not os.path.exists(file_path):
        print("文件不存在")
        exit(1)
    # for file in os.listdir(file_path):#list.dir可以访问所有的文件名
    #     print(file)
    files_name = sorted([file.split(".")[0] for file in os.listdir(file_path)])
    # 这句代码的意思就是遍历根目录下所有文件，然后用.进行分割，最后取文件名称
    # print(file_name)
    file_num = len(files_name)
    # print(file_num)#一共有2773张图象
    train_index = random.sample(range(0, file_num), k=int(file_num * train_rate))
    train_file = []
    val_file = []
    for index, file_name in enumerate(files_name):
        if index in train_index:
            train_file.append(file_name)
        else:
            val_file.append(file_name)
    try:
        train_f = open("train1.txt", "x")
        eval_f = open("val1.txt", "x")
        train_f.write("\n".join(train_file))
        eval_f.write("\n".join(val_file))
    except FileExistsError as e:
        print(e)
        exit(1)
split_data(file_path=file_path,train_rate=train_rate)

