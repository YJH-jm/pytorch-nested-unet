import numpy as np
import cv2
from PIL import Image

def blend_images(ori, pred):
    ori = cv2.cvtColor(ori, cv2.COLOR_BGR2RGB)
    output = Image.fromarray(pred)
    background = Image.fromarray(ori).convert('RGBA')
    output = output.resize((ori.shape[1], ori.shape[0])).convert('RGBA')
    output_final = Image.blend(background, output, alpha=0.5)
    return cv2.cvtColor(np.array(output_final), cv2.COLOR_BGR2RGB)