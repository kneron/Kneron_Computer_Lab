"""

Name    : voc2yolo.py
Author  : Oscar Law

Description:

    Convert labelImg Pascal/VOC format to Yolo absolute coordinate for training

History:
Aug 13, 2020. Oscar Law
- created
Aug 14, 2020. Oscar Law
- read the classes.txt
    
Copyright (c) 2020 Oscar Law
All Rights Reserved.
"""

import xml.etree.ElementTree as ET
import argparse
import glob
import os

parser = argparse.ArgumentParser(description="Convert VOC to Yolo format for training")
parser.add_argument('-c', '--class_list', default="classes.txt", help=("yolo class list"))
args = parser.parse_args()
classes = open(args.class_list).read().strip().split("\n")

train_list = open('train.txt', 'w')

for xml_name in glob.glob('*.xml'):
    xml_list = open(xml_name)
    tree = ET.parse(xml_list)
    root = tree.getroot()

    file_name = root.find('filename').text
    train_list.write("training/" + file_name)
    for obj in root.iter('object'):
        cls = obj.find('name').text
        cls_id   = classes.index(cls)
        xml_box  = obj.find('bndbox')
        xml_list = (int(xml_box.find('xmin').text), int(xml_box.find('ymin').text), int(xml_box.find('xmax').text), int(xml_box.find('ymax').text))
        train_list.write(" " + ",".join([str(xml_ptr) for xml_ptr in xml_list]) + ',' + str(cls_id))
    train_list.write('\n')
    
train_list.close()
