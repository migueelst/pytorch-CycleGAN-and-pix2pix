"""Post-hoc evaluation of model runs using clean-fid.

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from pathlib import Path
from options.eval_options import EvalOptions
import subprocess
import random
import shutil
from cleanfid import fid

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


if __name__ == '__main__':
    opt = EvalOptions().parse()  # get eval options

    # 1. Read  checkpoint dir to find all available generator checkpoints
    checkpoints_dir = Path(opt.checkpoints_dir)
    epochs = {int(n) for file in os.listdir(checkpoints_dir / opt.name)
              if (n := file.split("_")[0]).isdigit()}
    # Folder for FID calculation
    tmp_dir = Path(f"fid_{random.randint(0, 10000)}")
    for epoch in sorted(list(epochs)):
        print(f"Calculating FID for epoch {epoch}")
        # 2. Call test.py using subproc, set epoch to be each of the numbers found in 1), the last call should be using "latest"
        subprocess.run(["python", "test.py", "--dataroot", opt.dataroot, "--epoch", str(epoch),
                        "--name", opt.name, "--model", opt.model, "--dataset_mode", opt.dataset_mode,
                        "--direction", opt.direction, "--load_size", str(opt.load_size), "--crop_size", str(opt.crop_size)])
        # 3. Move all fake images and real images to a temporary folder
        fake_dir = tmp_dir / f"epoch_{epoch}" / "fake"
        real_dir = tmp_dir / f"epoch_{epoch}" / "real"
        fake_dir.mkdir(parents=True)
        real_dir.mkdir(parents=True)
        test_images_dir = Path(opt.results_dir) / opt.name / f"test_{epoch}" / "images" 
        for img_name in os.listdir(test_images_dir):
            if "_fake_B" in img_name:
                shutil.copy(test_images_dir / img_name, fake_dir / img_name)
            elif "_real_B" in img_name:
                shutil.copy(test_images_dir / img_name, real_dir / img_name)
        # 4. Calculate the clean_fid between both folders
        fid = fid.compute_fid(fake_dir, real_dir)
        print(f"FID for epoch {epoch}: {fid}")
        print(f"Calculating KID for epoch {epoch}")
        kid = fid.compute_kid(fake_dir, real_dir)
        print(f"KID for epoch {epoch}: {fid}")
        # 5. Log results to wandb
        print("--------------------------------")