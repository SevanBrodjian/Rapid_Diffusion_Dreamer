import argparse, os, sys
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
from imwatermark import WatermarkEncoder
import time

sys.path.append('c:\\Users\\Owner\\Documents\\image_generation\\rapid_diffusion_dreamer')

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

# ESRGAN Imports
import os.path as osp
import glob
import RRDBNet_arch as arch


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    if device == torch.device("cuda"):
        model.cuda()
    elif device == torch.device("cpu"):
        model.cpu()
        model.cond_stage_model.device = "cpu"
    else:
        raise ValueError(f"Incorrect device name. Received: {device}")
    model.eval()
    return model


def init_sd():
    config = OmegaConf.load("../configs/stable-diffusion/v2-inference.yaml")
    device = torch.device("cuda")
    return load_model_from_config(config, "../model_weights/v2-1_512-ema-pruned.ckpt", device)


def initialize_esrgan(esrgan_model_path = './esrgan_models/RRDB_ESRGAN_x4.pth'):
    esrgan_device = torch.device('cuda')
    esrgan_model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    esrgan_model.load_state_dict(torch.load(esrgan_model_path), strict=True)
    esrgan_model.eval()
    return esrgan_model.to(esrgan_device)


def upscale_images(raw_imgs, esrgan_model):
    start_time = time.time()
    raw_imgs = torch.cat(raw_imgs, dim=0).to(torch.float32)
    with torch.no_grad():
        images = esrgan_model(raw_imgs).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if len(images.shape) == 3: # Single image (add batch dimension for consistency)
        images = np.array([images])
    images = np.transpose(images, (0, 2, 3, 1))
    images = (images * 255.0).round()
    end_time = time.time()
    return images, (end_time - start_time)


def get_options_dict(model, steps = 50, h = 576, w = 1024, rand_seed = True, outpath = 'sandbox', plms = False, dpm = False):
    options = {}
    options['device'] = torch.device("cuda")
    if plms:
        options['sampler'] = PLMSSampler(model, device=options['device'])
    elif dpm:
        options['sampler'] = DPMSolverSampler(model, device=options['device'])
    else:
        options['sampler'] = DDIMSampler(model, device=options['device'])
    options['outpath'] = '../outputs/' + outpath
    options['n_samples']= 1
    options['batch_size'] = 1
    options['n_rows'] = 1
    options['sample_path'] = os.path.join(options['outpath'], "samples")
    os.makedirs(options['sample_path'], exist_ok=True)
    options['sample_count'] = 0
    options['base_count'] = len(os.listdir(options['sample_path']))
    options['grid_count'] = len(os.listdir(options['outpath'])) - 1
    options['start_code'] = None
    options['precision_scope'] = autocast
    options['sampler'] = DDIMSampler(model, device=options['device'])
    options['scale'] = 9
    options['opt_C'] = 4
    options['opt_H'] = h
    options['opt_f'] = 8
    options['opt_W'] = w
    options['steps'] = steps
    options['ddim_eta'] = 0.0
    if rand_seed:
        options['seed'] = np.random.randint(9999999)
    else:
        options['seed'] = 777
    return options


def encode_text(model, prompt, options):
    seed_everything(options['seed'])
    data = [options['batch_size'] * [prompt]]

    start_time = time.time()
    with torch.no_grad(), \
        options['precision_scope']("cuda"), \
        model.ema_scope():
            conditionings = list()
            unconditional_conditionings = list()
            for n in trange(options['n_samples'], desc="Sampling"):
                for prompts in tqdm(data, desc="data"):
                    uc = None
                    if options['scale'] != 1.0:
                        uc = model.get_learned_conditioning(options['batch_size'] * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts)
                    unconditional_conditionings.append(uc)
                    conditionings.append(c)
    end_time = time.time()
    return unconditional_conditionings, conditionings, (end_time - start_time)


def encode_samples(model, unconditional_conditionings, conditionings, options):
    opt_C = options['opt_C']
    opt_H = options['opt_H']
    opt_f = options['opt_f']
    opt_W = options['opt_W']
    shape = [opt_C, opt_H // opt_f, opt_W // opt_f]

    start_time = time.time()
    with torch.no_grad(), \
        options['precision_scope']("cuda"), \
        model.ema_scope():
            encoded_samples = list()
            for uc, c in list(zip(unconditional_conditionings, conditionings)):
                samples, _ = options['sampler'].sample(S=options['steps'],
                                                    conditioning=c,
                                                    batch_size=options['batch_size'],
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=options['scale'],
                                                    unconditional_conditioning=uc,
                                                    eta=options['ddim_eta'],
                                                    x_T=options['start_code'])
                encoded_samples.append(samples)
    end_time = time.time()

    return encoded_samples, (end_time - start_time)


def decode_imgs(model, encoded_samples, options):
    start_time = time.time()
    with torch.no_grad(), \
        options['precision_scope']("cuda"), \
        model.ema_scope():
            all_samples = list()
            for samples in encoded_samples:
                    x_samples = model.decode_first_stage(samples)
                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                    all_samples.append(x_samples)
    end_time = time.time()

    return all_samples, (end_time - start_time)


def save_images(images, options, sample_path = 'samples'):
    start_time = time.time()
    sample_path = os.path.join(options['outpath'], sample_path)
    os.makedirs(sample_path, exist_ok=True)
    for img in images:
        base_count = len(os.listdir(sample_path))
        img = Image.fromarray(img.astype(np.uint8))
        img.save(os.path.join(sample_path, f"{base_count:05}.png"))
    end_time = time.time()

    return (end_time - start_time)