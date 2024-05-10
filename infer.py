"""Script for generating depth and normal inferences from Metric3D.

Arguments:
    --cfg: path to the Metric3D model configuration file
    --weights: path to the .pth checkpoint of the model
    --data: path to the dataset annotations (.json) file
    --out: the output directory to store inferences
"""

import os
import sys
import torch
import cv2
from tqdm import tqdm
import argparse
from torchvision.utils import save_image

CODE_SPACE=os.path.abspath('./Metric3D')
sys.path.append(CODE_SPACE)
os.chdir(CODE_SPACE) 

from mono.utils.running import load_ckpt
from mono.model.monodepth_model import get_configured_monodepth_model
from mono.utils.custom_data import load_from_annos, load_data
from mono.utils.do_test import transform_test_data_scalecano, get_prediction

try:
    from mmcv.utils import Config, DictAction
except:
    from mmengine import Config, DictAction
    

def parse_args():
    """Parse the command line arguments."""
    
    # Create the parser
    parser = argparse.ArgumentParser(description="Process the command line arguments for inference.")

    # Add arguments
    parser.add_argument('--cfg', type=str, required=True, help='Path to the model configuration file (.py)')
    parser.add_argument('--weights', type=str, required=True, help='Path to the weights file (.pth)')
    parser.add_argument('--data', type=str, required=True, help='Path to the data annotations file (.json)')
    parser.add_argument('--out', type=str, required=True, help='Path to the output directory')

    # Parse arguments
    args = parser.parse_args()
    
    return args


def get_absolute_path(base_dir, relative_path):
    """Return the absolute path of relative_path, relative to base_dir."""
    
    # Join the base directory and the relative path
    combined_path = os.path.join(base_dir, relative_path)
    
    # Convert the combined path to an absolute path
    absolute_path = os.path.abspath(combined_path)
    
    return absolute_path
    
    

def load_model(cfg: Config, weights: str) -> torch.nn.Module:
    """Load a Metric3D checkpoint given a configuration and weights file path."""
    
    # Load model
    model = get_configured_monodepth_model(cfg, )
    model = torch.nn.DataParallel(model).float().cuda() # Keep this even when using one GPU, or Metric3D modules break

    model, _,  _, _ = load_ckpt(weights, model, strict_match=False)
    model.eval()

    return model


def infer_depth_and_normal(model: torch.nn.Module, an: dict, normalize_scale: tuple[float, float]):
    """Infer the depth and normal, given a single example annotation.
    
    Args:
        - model: the model used for inference
        - an: the single example annotation
        - normalize_scale: the depth range for the model
    """
    
    rgb_origin = cv2.imread(an['rgb'])[:, :, ::-1].copy()
    
    if an['depth'] is not None:
        gt_depth = cv2.imread(an['depth'], -1)
        gt_depth_scale = an['depth_scale']
        gt_depth = gt_depth / gt_depth_scale
    else:
        gt_depth = None
        
    intrinsic = an['intrinsic']
    if intrinsic is None:
        intrinsic = [1000.0, 1000.0, rgb_origin.shape[1]/2, rgb_origin.shape[0]/2]

    rgb_input, cam_models_stacks, pad, label_scale_factor = transform_test_data_scalecano(rgb_origin, intrinsic, cfg.data_basic)

    pred_depth, _, _, output = get_prediction(
        model = model,
        input = rgb_input,
        cam_model = cam_models_stacks,
        pad_info = pad,
        scale_info = label_scale_factor,
        gt_depth = None,
        normalize_scale = normalize_scale,
        ori_shape=[rgb_origin.shape[0], rgb_origin.shape[1]],
    )
    
    pred_depth = (pred_depth > 0) * (pred_depth < 300) * pred_depth
    
    normal_out_list = output['normal_out_list'] 
    pred_normal = normal_out_list[0][:, :3, :, :] # (B, 3, H, W)
    H, W = pred_normal.shape[2:]
    pred_normal = pred_normal[:, :, pad[0]:H-pad[1], pad[2]:W-pad[3]]

    return pred_depth, pred_normal


def get_filename(path: str):
    """Return the filename of a path, without any extensions"""
    
    base_name = os.path.basename(path)
    
    return os.path.splitext(base_name)[0]    


if __name__ == '__main__':
    args = parse_args()

    data = load_from_annos(get_absolute_path('..', args.data))
    cfg = Config.fromfile(get_absolute_path('..', args.cfg))
    model = load_model(cfg, get_absolute_path('..', args.weights))

    normalize_scale = cfg.data_basic.depth_range[1]
    
    out = get_absolute_path('..', args.out)
    os.makedirs(out, exist_ok=True)
    
    for i, an in tqdm(enumerate(data)):
        pred_depth, pred_normal = infer_depth_and_normal(model, an, normalize_scale)

        torch.save(pred_depth.cpu(), os.path.join(out, f'{get_filename(data[i]["rgb"])}_depth.pt'))
        torch.save(pred_normal.cpu(), os.path.join(out, f'{get_filename(data[i]["rgb"])}_normal.pt'))
