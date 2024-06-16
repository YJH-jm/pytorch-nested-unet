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

    save_path = "./inputs/Kvasir-SEG"
    
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # 마스크 이동
    shutil.copytree(os.path.join(args.data_dir, 'masks'), os.path.join(save_path, 'masks/0'))



   # 이미지 이동 
    shutil.copytree(os.path.join(args.data_dir, 'images'), os.path.join(save_path, 'images/'))

    
    
if __name__ == "__main__":
    args = parse_arguments()
    main(args)

