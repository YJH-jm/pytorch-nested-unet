import os
import sys
import shutil
import argparse
import numpy as np
import cv2

def parse_arguments():
    parser = argparse.ArgumentParser(description="Data Preprocessing ")
    parser.add_argument("data_dir", type=str, help="data download path")
    
    return parser.parse_args()


def main(args):

    if not os.path.isdir(args.data_dir):
        print("Data folder dose not exist")
        sys.exit()

    save_path = "./inputs/Lane_Seg"
    
    masks_path = ['masks/0', 'masks/1']
    label_path  = os.path.join(args.data_dir, "train_label")


    if not os.path.exists(os.path.join(save_path, 'masks/0')):
        os.makedirs(os.path.join(save_path, 'masks/0'))
    
    if not os.path.exists(os.path.join(save_path, 'masks/1')):
        os.makedirs(os.path.join(save_path, 'masks/1'))

    
    # mask label 별로 나누기 
    label_files = os.listdir(label_path)
    
    for file in label_files:
        img = cv2.imread(os.path.join(label_path,file))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mask_l = np.where(gray_img==1, 255, 0)
        mask_r = np.where(gray_img==2, 255, 0)
        
        cv2.imwrite(os.path.join(save_path, 'masks/0', file), mask_l)
        cv2.imwrite(os.path.join(save_path, 'masks/1', file), mask_r)


    # label 이름 수정, 학습 이미지와 같게 만들어줌
    for mask_path in masks_path:
        label_files = os.listdir(os.path.join(save_path, mask_path))
        for file in label_files:
            new_name = file.replace("_label","")
            os.rename(os.path.join(save_path, mask_path, file), os.path.join(save_path, mask_path, new_name))
    
    # image 파일 옮기기
    shutil.copytree(os.path.join(args.data_dir, 'train'), os.path.join(save_path, 'images/'))
    
    
if __name__ == "__main__":
    args = parse_arguments()
    main(args)
