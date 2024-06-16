import os
import argparse
import yaml
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

import archs
from vis import blend_images

def parse_arguments():
    parser = argparse.ArgumentParser(description="Inference ")
    parser.add_argument("model", type=str, help="trained model path")
    parser.add_argument("config", type=str, help="trained model config file path")
    parser.add_argument("sample_images", type=str, help="sample image folder")
    parser.add_argument("sample_gt", type=str, help="sample gt image folder")
    # parser.add_argument("--age", type=int, help="나이를 입력하세요", required=True)
    
    return parser.parse_args()



def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    model = archs.__dict__[cfg['arch']](cfg['num_classes'],cfg['input_channels'],cfg['deep_supervision'])
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device)

    img_folder = args.sample_images
    gt_folder = args.sample_gt


    result_folder = os.path.join("./outputs", cfg['name'] + "_infer")
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    img_list = sorted(os.listdir(img_folder))
    gt_list = sorted(os.listdir(gt_folder))


    for idx, (img_file, gt_file) in enumerate(zip(img_list,gt_list)):
        black = np.zeros(shape=(224,224*4,3), dtype=np.uint8)
        gt = cv2.imread(os.path.join(gt_folder, gt_file))
        print(os.path.join(gt_folder, gt_file))
        gt = cv2.resize(gt, (224,224)) # (384, 384, 3)
        
        img = cv2.imread(os.path.join(img_folder, img_file), cv2.IMREAD_COLOR)
        print(os.path.join(img_folder, img_file))
        img = cv2.resize(img, (224,224)) # (384, 384, 3)
        input = img.astype('float32') / 255

       
        input = np.expand_dims(input, axis=0) # (1, 384, 384, 3)
        input = torch.from_numpy(input).to(device) # (1, 384, 384, 3)
        input = input.permute(0,3,1,2)
        output = model(input) # torch.Size([1, 3, 384, 384])
        output = torch.sigmoid(output) # torch.Size([1, 1, 384, 384])
        output = output.permute(0,2,3,1).cpu().detach() # torch.Size([1, 384, 384, 1])
        pred = np.array(output[0])*255 
        # pred = np.where(pred<240, 0, pred)
        pred = np.where(pred<99, 0, 255)
        pred_ = np.repeat(pred, 3, -1).astype(np.uint8)
        output_final = blend_images(img, pred_)[:,:,:3]
        
        cv2.putText(img, "Origninal Image", (70,40),cv2.FONT_HERSHEY_DUPLEX, 1,(0,0,255), thickness=3, lineType=cv2.LINE_AA)
        cv2.putText(gt, "GroundTruth Mask", (60,40),cv2.FONT_HERSHEY_DUPLEX, 1,(0,0,255), thickness=3, lineType=cv2.LINE_AA)
        cv2.putText(pred_, "Predicted Mask", (70,40),cv2.FONT_HERSHEY_DUPLEX, 1,(0,0,255), thickness=3, lineType=cv2.LINE_AA)
        cv2.putText(output_final, "Blended Images", (60,40),cv2.FONT_HERSHEY_DUPLEX, 1,(0,0,255), thickness=3, lineType=cv2.LINE_AA)
        
        black[:,:224,:] = img[:,:,:]
        black[:,224:224*2,:] = gt[:,:,:]
        black[:,224*2:224*3,:] = pred_[:,:,:]
        black[:,224*3:224*4,:] = output_final[:,:,::]

        cv2.imwrite(os.path.join(result_folder, img_file), black)
        
        
        if idx <10:
            plt.imshow(cv2.cvtColor(black, cv2.COLOR_BGR2RGB))
            plt.show()
        
        if idx == 99:
            break
if __name__ == "__main__":
    args = parse_arguments()
    main(args)