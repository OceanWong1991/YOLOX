# -*- encoding: utf-8 -*-
'''
/* ***************************************************************************************************
*   NOTICE
*   This software is the property of Glint Co.,Ltd.. Any information contained in this
*   doc should not be reproduced, or used, or disclosed without the written authorization from
*   Glint Co.,Ltd..
***************************************************************************************************
*   File Name       : train_val_split.py
***************************************************************************************************
*    Module Name        : 
*    Prefix            : 
*    ECU Dependence    : None
*    MCU Dependence    : None
*    Mod Dependence    : None
***************************************************************************************************
*    Description        : 
*
***************************************************************************************************
*    Limitations        :
*
***************************************************************************************************
*
***************************************************************************************************
*    Revision History:
*
*    Version        Date            Initials        CR#                Descriptions
*    ---------    ----------        ------------    ----------        ---------------
*     1.0.0       2023-11-09            Neo                         
****************************************************************************************************/
'''
import os
import glob
import shutil
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split

def check_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def gen_train_val_list_with_path(src, des):
    check_mkdir(des)
    path = Path(src)
    files = glob.glob(str(path / '**' / 'images' / '*.jpg'), recursive=True)

    # OPT change test size if you want to reset tran val data sise
    X_train, X_val = train_test_split(files, test_size = 0.12, random_state = 1991)
    print('train data set length....', len(X_train))
    print('val data set length....', len(X_val))

    with open(os.path.join(des, 'train.txt'), 'w') as f1:
        for i in X_train:
            f1.write(i + '\n')

    with open(os.path.join(des, 'val.txt'), 'w') as f1:
        for i in X_val:
            f1.write(i + '\n')

def gen_train_val_list(src, des):
    check_mkdir(des)
    path = Path(src)
    files = glob.glob(str(path / 'images' / '*.jpg'), recursive=True)

    # OPT change test size if you want to reset tran val data sise
    X_train, X_val = train_test_split(files, test_size = 0.00002, random_state = 1991)
    print('train data set length....', len(X_train))
    print('val data set length....', len(X_val))

    with open(os.path.join(des, 'train.txt'), 'w') as f1:
        for i in X_train:
            i = i.split(".jpg")[0].strip().split('/')[-1]
            f1.write(i + '\n')

    with open(os.path.join(des, 'val.txt'), 'w') as f1:
        for i in X_val:
            i = i.split(".jpg")[0].strip().split('/')[-1]
            f1.write(i + '\n')

def moveFile(source, des, type):
    save_path = os.path.join(des, type)
    check_mkdir(save_path)

    with open(os.path.join(source, 'train.txt'), 'r') as f:
        file = f.readlines()
        for p in tqdm(file):
            p = p.strip()
            imgname = os.path.basename(p)
            txt = p.rsplit('.')[0] + '.txt'
            txtname = os.path.basename(txt)
            shutil.copyfile(p, os.path.join(des, imgname))
            shutil.copyfile(txt, os.path.join(des, txtname))

if __name__ == '__main__':
    src = '/home/glint/xzwang/data/0327'
    des = '/home/glint/xzwang/data/0327'
    gen_train_val_list_with_path(src, des)
