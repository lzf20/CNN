from torch.utils.data import Dataset
import os
import torch
import json
from PIL import Image
from lxml import etree
class landslideDataSet(Dataset):
    """读取滑坡数据集"""
    #init函数的目的就是得到图像的路径，然后将图像路径组成一个数组，这样在getitem种就可以直接读取
    def __init__(self,slide_root,transforms,train_set=True):
        self.root=slide_root
        self.img_root="./data1"
        #读取train.txt或者val.txt文件
        if train_set:
            txt_list=os.path.join(self.root,"train.txt")
        else:
            txt_list=os.path.join(self.root,"val.txt")
        with open(txt_list) as read:
            self.xml_list=[os.path.join(self.annotations_root,line.strip()+".xml")
                           for line in read.readlines() ]
        try:
            json_file=open('label.json','r')
            self.class_dict=json.load(json_file)
        except Exception as e:
            print(e)
            exit(-1)
        self.transforms=transforms
    def __len__(self):
        return len(self.xml_list)
    def __getitem__(self, idx):
        #读xml文件
        xml_path=self.xml_list[idx]
        with open(xml_path) as fid:
            xml_str=fid.read()
        xml=etree.fromstring(xml_str)
        data=self.parse_xml_to_dict(xml)[""]

