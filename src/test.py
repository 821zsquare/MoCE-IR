import os
import pathlib
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import List
from skimage import img_as_ubyte
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.utils.data import DataLoader

from net.moce_ir import MoCEIR
from net.moe_ir import MoEIR
from net.msdr_moe_ir import MSDRMoEIR
from net.msdr_stage_moe_ir import MSDRStageMoEIR
from options import train_options
from utils.test_utils import save_img
from utils.expert_stats import ExpertActivationStats
from data.dataset_utils import IRBenchmarks, CDD11


BENCHMARK_RUNNERS = {
    "synllie": "lolv1",
    "deblur": "gopro",
}


def resolve_output_root(opt):
    """Resolve root folder for saved outputs.

    Keep backward compatibility:
    - default --output_path ("output/") -> save under current working directory.
    - custom --output_path -> use that path as root.
    """
    output_path = getattr(opt, "output_path", None)
    if output_path is None:
        return os.getcwd()
    normalized = os.path.normpath(os.path.expanduser(output_path))
    if normalized in {"output", "."}:
        return os.getcwd()
    return normalized


def build_net(opt):
    common_kwargs = dict(
        dim=opt.dim,
        num_blocks=opt.num_blocks,
        num_dec_blocks=opt.num_dec_blocks,
        levels=len(opt.num_blocks),
        heads=opt.heads,
        num_refinement_blocks=opt.num_refinement_blocks,
        topk=opt.topk,
        num_experts=opt.num_exp_blocks,
        rank=opt.latent_dim,
        stage_depth=opt.stage_depth,
    )
    if opt.model in {"MoE_IR", "MoE_IR_S"}:
        return MoEIR(**common_kwargs)
    if opt.model in {"MSDR_MoE_IR", "MSDR_MoE_IR_S"}:
        return MSDRMoEIR(**common_kwargs)
    if opt.model in {"MSDR_Stage_MoE_IR", "MSDR_Stage_MoE_IR_S"}:
        return MSDRStageMoEIR(**common_kwargs)
    return MoCEIR(
        **common_kwargs,
        with_complexity=opt.with_complexity,
        depth_type=opt.depth_type,
        rank_type=opt.rank_type,
        complexity_scale=opt.complexity_scale,
    )



####################################################################################################
## HELPERS
def compute_psnr(image_true, image_test, image_mask, data_range=None):
  # this function is based on skimage.metrics.peak_signal_noise_ratio
  err = np.sum((image_true - image_test) ** 2, dtype=np.float64) / np.sum(image_mask)
  return 10 * np.log10((data_range ** 2) / err)

def compute_ssim(tar_img, prd_img, cr1):
    ssim_pre, ssim_map = structural_similarity(tar_img, prd_img, channel_axis=2, gaussian_weights=True, data_range = 1.0, full=True)
    ssim_map = ssim_map * cr1
    r = int(3.5 * 1.5 + 0.5)  # radius as in ndimage
    win_size = 2 * r + 1
    pad = (win_size - 1) // 2
    ssim = ssim_map[pad:-pad,pad:-pad,:]
    crop_cr1 = cr1[pad:-pad,pad:-pad,:]
    ssim = ssim.sum(axis=0).sum(axis=0)/crop_cr1.sum(axis=0).sum(axis=0)
    ssim = np.mean(ssim)
    return ssim

def calc_psnr(img1, img2, data_range=1.0):
    err = np.sum((img1 - img2) ** 2, dtype=np.float64)
    return 10 * np.log10((data_range ** 2) / (err / img1.size))

def calc_ssim(img1, img2):
    return structural_similarity(img1, img2, channel_axis=2, gaussian_weights=True, data_range = 1.0, full=False)


def _scalar_value(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().view(-1).tolist()
    if isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return _scalar_value(value[0])
        return "|".join(str(_scalar_value(item)) for item in value)
    return value


def _degradation_label(de_id, opts):
    value = _scalar_value(de_id)
    de_types = getattr(opts, "de_type", None) or []
    if isinstance(value, (int, np.integer)) and 0 <= int(value) < len(de_types):
        return de_types[int(value)]
    return str(value)


def _write_metric_csvs(result_dir, detail_rows, psnr, ssim, lpips):
    detail_path = os.path.join(result_dir, "results-detail.csv")
    overview_path = os.path.join(result_dir, "results-overview.csv")

    with open(detail_path, "w", newline="") as f:
        fieldnames = ["image", "degradation", "psnr", "ssim", "lpips", "saved_image"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)

    overview_rows = []
    for metric, values in (("psnr", psnr), ("ssim", ssim), ("lpips", lpips)):
        values = np.asarray(values, dtype=np.float64)
        overview_rows.append({
            "metric": metric,
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        })

    with open(overview_path, "w", newline="") as f:
        fieldnames = ["metric", "mean", "min", "max"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(overview_rows)

    return detail_path, overview_path



####################################################################################################
## PL Test Model
class PLTestModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()

        self.net = build_net(opt)
    
    def forward(self,x):
        return self.net(x)


####################################################################################################
def run_test(opts, net, dataset, factor=8, expert_stats=None):
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, drop_last=False, num_workers=0)

    output_root = resolve_output_root(opts)
    result_dir = os.path.join(output_root, "results", opts.checkpoint_id, opts.benchmarks[0])
    pathlib.Path(result_dir).mkdir(parents=True, exist_ok=True)
    calc_lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True, reduction="mean").cuda()
    psnr, ssim, lpips = [], [], []
    detail_rows = []
    with torch.no_grad():

        for ([clean_name, de_id], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()
                        
            # Forward pass
            if expert_stats is not None:
                expert_stats.set_batch(de_id)
            restored = net(degrad_patch)
            if isinstance(restored, List) and len(restored) == 2:
                restored , _ = restored
            
            # Unpad images to original dimensions
            assert restored.shape == clean_patch.shape, "Restored and clean patch shape mismatch."
            
            # save output images
            restored = torch.clamp(restored,0,1)
            lpips_temp = float(calc_lpips(clean_patch, restored).detach().cpu().item())
            lpips.append(lpips_temp)
            
            restored = restored.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            degrad_patch = degrad_patch.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            clean = clean_patch.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            ssim_temp = float(calc_ssim(clean, restored))
            ssim.append(ssim_temp)
            psnr_temp = peak_signal_noise_ratio(clean, restored, data_range=1)
            psnr_temp = float(psnr_temp)
            psnr.append(psnr_temp)

            image_path = str(_scalar_value(clean_name))
            degradation = _degradation_label(de_id, opts)
            saved_image = ""
            if opts.save_results:
                image_stem = os.path.splitext(os.path.basename(image_path))[0]
                save_name = f"{degradation}_{image_stem}_{round(psnr_temp, 2)}.png"
                saved_image = os.path.join(result_dir, save_name)
                save_img(saved_image, img_as_ubyte(restored))

            detail_rows.append({
                "image": image_path,
                "degradation": degradation,
                "psnr": psnr_temp,
                "ssim": ssim_temp,
                "lpips": lpips_temp,
                "saved_image": saved_image,
            })

    detail_path, overview_path = _write_metric_csvs(result_dir, detail_rows, psnr, ssim, lpips)
    print('PSNR: {:f} SSIM: {:f} LPIPS: {:f}\n'.format(np.mean(psnr), np.mean(ssim), np.mean(lpips)))
    print(f"Metric details saved to: {detail_path}")
    print(f"Metric overview saved to: {overview_path}")

            
## test LolV1
def run_lolv1(opts, net, dataset, factor=8, expert_stats=None):
    run_test(opts, net, dataset, factor, expert_stats)


def run_synllie(opts, net, dataset, factor=8, expert_stats=None):
    run_lolv1(opts, net, dataset, factor, expert_stats)
    
## test GoPro
def run_gopro(opts, net, dataset, factor=8, expert_stats=None):
    run_test(opts, net, dataset, factor, expert_stats)


def run_deblur(opts, net, dataset, factor=8, expert_stats=None):
    run_gopro(opts, net, dataset, factor, expert_stats)


def run_driving(opts, net, dataset, factor=8, expert_stats=None):
    run_test(opts, net, dataset, factor, expert_stats)
        
## test Derain
def run_derain(opts, net, dataset, factor=8, expert_stats=None):
    run_test(opts, net, dataset, factor, expert_stats)
        
## test Dehaze
def run_dehaze(opts, net, dataset, factor=8, expert_stats=None):
    run_test(opts, net, dataset, factor, expert_stats)
    
## test synthetic denoising
def run_denoise_15(opts, net, dataset, factor=8, expert_stats=None):
    run_test(opts, net, dataset, factor, expert_stats)
    
def run_denoise_25(opts, net, dataset, factor=8, expert_stats=None):
    run_test(opts, net, dataset, factor, expert_stats)
    
def run_denoise_50(opts, net, dataset, factor=8, expert_stats=None):
    run_test(opts, net, dataset, factor, expert_stats)


def run_denoise_75(opts, net, dataset, factor=8, expert_stats=None):
    run_test(opts, net, dataset, factor, expert_stats)


def run_denoise_100(opts, net, dataset, factor=8, expert_stats=None):
    run_test(opts, net, dataset, factor, expert_stats)

# test CDD11
def run_cdd11(opts, net, dataset, factor=8, expert_stats=None):
    run_test(opts, net, dataset, factor, expert_stats)


####################################################################################################
## main
def main(opt):
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Load model
    net = PLTestModel.load_from_checkpoint(
        os.path.join(opt.ckpt_dir, opt.checkpoint_id, "last.ckpt"), opt=opt).cuda()
    net.eval()

    expert_stats = None
    if opt.save_expert_stats:
        expert_stats = ExpertActivationStats(net, degradation_names=opt.de_type)
        print(f"Collecting expert activation stats from {len(expert_stats.layers)} decoder adapter layers.")
    
    for de in list(opt.benchmarks):
        ind_opt = argparse.Namespace(**vars(opt))
        ind_opt.benchmarks = [de]
        
        if "CDD11" in ind_opt.trainset:
            _, subset = ind_opt.trainset.split("_", maxsplit=1)
            dataset = CDD11(ind_opt, split="test", subset=subset)
        else:
            dataset = IRBenchmarks(ind_opt)
        
        print("--------> Testing on", de, "testset.")
        print("\n")
        runner_name = BENCHMARK_RUNNERS.get(de, de)
        runner = globals().get(f"run_{runner_name}")
        if runner is None:
            raise NotImplementedError(f"Benchmark runner for '{de}' not found.")
        runner(ind_opt, net, dataset, factor=8, expert_stats=expert_stats)

    if expert_stats is not None:
        stats_dir = os.path.join(resolve_output_root(opt), "results", opt.checkpoint_id)
        json_path, csv_path = expert_stats.save(stats_dir)
        expert_stats.close()
        print(f"Expert activation stats saved to:\n  {json_path}\n  {csv_path}")
    

def depth_type(value):
    try:
        return int(value)  # Try to convert to int
    except ValueError:
        return value  # If it fails, return the string
    
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
    
    
if __name__ == '__main__':
    train_opt = train_options()
    main(train_opt)
