"""Post-hoc evaluation of model runs using clean-fid.

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from pathlib import Path
from collections.abc import Callable
from options.eval_options import EvalOptions
import subprocess
import random
import shutil
from cleanfid.fid import kernel_distance, get_folder_features, build_feature_extractor, fid_from_feats
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


def parse_config_file(file_path: str):
    config = {}
    
    with open(file_path, 'r') as file:
        for line in file:
            if ":" not in line:
                continue
            key, value = line.split(':', 1) 
            key = key.strip()
            value = value.split('[')[0].strip()  # Remove default values in brackets
            config[key] = value
    
    return config


def calculate_features_from_folder(folder_path: str, feat_model, img_transform: Callable = None):
    # Use custom transformations for FID
    # TODO: get num workers from command line
    if img_transform:
        feats = get_folder_features(folder_path, model=feat_model, num_workers=8, num=None,
                shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
                mode="clean", custom_image_tranform=img_transform, description="", verbose=True)
    else:
        feats = get_folder_features(folder_path, model=feat_model, num_workers=8, num=None,
                shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
                mode="clean",  description="", verbose=True)
    return feats

def kid_from_feats(feats1: np.array, feats2: np.array):
    return kernel_distance(feats1, feats2)


if __name__ == '__main__':
    opt = EvalOptions().parse()  # TODO: add argument to cleanup files
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config={})
        wandb_run._label(repo='CycleGAN-and-pix2pix')
    # We want to log the params that were used to train the model, not the ones being used to call the evaluation script
    checkpoints_dir = Path(opt.checkpoints_dir)
    if opt.use_wandb:
        wandb_run.config.update = parse_config_file(str(checkpoints_dir / opt.name / "train_opt.txt"))
        wandb_run.define_metric("fid", summary="min")
        wandb_run.define_metric("kid", summary="min")
    # Folder for FID calculation
    tmp_dir = Path(f"fid_{random.randint(0, 10000)}")
    epochs = {int(n) for file in os.listdir(checkpoints_dir / opt.name)
              if (n := file.split("_")[0]).isdigit()}
    sorted_epochs = sorted(list(epochs))
    for i, epoch in enumerate(sorted_epochs):
        print(f"Calculating FID for epoch {epoch}")
        # TODO: if I have time dont call test but rather inference directly here and store only the fake images in the fakes folder
        subprocess.run(["python", "test.py", "--dataroot", opt.dataroot, "--epoch", str(epoch),
                        "--name", opt.name, "--model", opt.model, "--dataset_mode", opt.dataset_mode,
                        "--direction", opt.direction, "--load_size", str(opt.load_size),
                        "--crop_size", str(opt.crop_size)], stdout=subprocess.DEVNULL)
        # TODO: this logic could be encapsulated in the metrics class
        fake_dir = tmp_dir / "fakes" / f"epoch_{epoch}"
        fake_dir.mkdir(exist_ok=True, parents=True)
        real_dir = tmp_dir / "real"
        real_dir.mkdir(exist_ok=True, parents=True)
        test_images_dir = Path(opt.results_dir) / opt.name / f"test_{epoch}" / "images" 
        for img_name in os.listdir(test_images_dir):
            if "_fake_B" in img_name:
                shutil.copy(test_images_dir / img_name, fake_dir / img_name)
            elif "_real_B" in img_name and i == 0:
                shutil.copy(test_images_dir / img_name, real_dir / img_name)
        if i == 0:
            # TODO: later on create a metrics base class and from that one create an FID class
            def fn_transform(x):
                x_pil = Image.fromarray(x)
                out_pil = transforms.Resize(opt.load_size, interpolation=transforms.InterpolationMode.LANCZOS)(x_pil)
                return np.array(out_pil)
            print("Getting features for real images for metric calculations")
            feature_extractor = build_feature_extractor("clean", "cuda", use_dataparallel=False)
            real_feats = calculate_features_from_folder(str(real_dir), feature_extractor, fn_transform)
        print("Calculating FID ...")
        fake_feats = calculate_features_from_folder(str(fake_dir), feature_extractor)
        # 4. Calculate the clean_fid between both folders
        fid = fid_from_feats(real_feats, fake_feats)
        print(f"Calculating KID ...")
        kid = kid_from_feats(real_feats, fake_feats) * 1000  # To make it easier to read
        print(f"FID for epoch {epoch}: {fid}")
        print(f"KID for epoch {epoch}: {kid}")
        # 5. Log results to wandb
        wandb_run.log({"epoch": epoch, "fid": fid, "kid": kid})