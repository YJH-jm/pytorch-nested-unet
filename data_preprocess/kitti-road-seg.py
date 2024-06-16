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

    save_path = "./inputs/KITTI_Road_Seg"
    label_path = os.path.join(args.data_dir, 'training', 'semantic')
  
    if not os.path.exists(os.path.join(save_path, 'masks', '0')):
        os.makedirs(os.path.join(save_path, 'masks', '0'))


    
    # mask label 별로 나누기 
    label_files = os.listdir(label_path)
    
    for file in label_files:
        img = cv2.imread(os.path.join(label_path, file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = np.where(img==7, 255, 0)
        cv2.imwrite(os.path.join(save_path, 'masks', '0', file), mask)


   # 이미지 이동 
    shutil.copytree(os.path.join(args.data_dir, 'training', 'image_2'), os.path.join(save_path, 'images/'))

    
    
if __name__ == "__main__":
    args = parse_arguments()
    main(args)
