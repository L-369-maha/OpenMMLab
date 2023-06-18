import os, cv2
import numpy as np
import mmcv
from mmengine import Config
from PIL import Image

from mmagic.registry import MODELS
from mmagic.utils import register_all_modules

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
register_all_modules()

cfg = Config.fromfile('./configs/controlnet/controlnet-canny.py')
controlnet = MODELS.build(cfg.model).cuda()

control_url = './project/dataset/p2.jpg'
control_img = mmcv.imread(control_url)
control = cv2.Canny(control_img, 100, 100)
control = control[:, :, None]
control = np.concatenate([control] * 3, axis=2)
control = Image.fromarray(control)
control.save('./project/dataset/p2_canny.png')

prompt = 'Room with blue walls and a yellow ceiling.'

output_dict = controlnet.infer(prompt, control=control)
samples = output_dict['samples']
for idx, sample in enumerate(samples):
    sample.save(f'sample_{idx}.png')
controls = output_dict['controls']
for idx, control in enumerate(controls):
    control.save(f'control_{idx}.png')