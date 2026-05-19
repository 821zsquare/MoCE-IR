from typing import List  # 导入类型注解中的 List（当前文件中暂未使用）

import os  # 导入操作系统相关工具，用于路径拼接等
import pathlib  # 导入 pathlib，用于跨平台路径与目录创建
import numpy as np  # 导入 numpy（当前文件中暂未使用）

from tqdm import tqdm  # 导入进度条工具 tqdm（当前文件中暂未使用）
from datetime import datetime  # 导入 datetime，用于生成时间戳

import torch.nn as nn  # 导入 PyTorch 神经网络模块
import torch
import torch.nn.functional as F
import torch.optim as optim  # 导入 PyTorch 优化器模块
import lightning.pytorch as pl  # 导入 PyTorch Lightning 主模块
from torch.utils.data import DataLoader  # 导入数据加载器
from lightning.pytorch.callbacks import ModelCheckpoint  # 导入模型保存回调
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger  # 导入 WandB 与 TensorBoard 日志器

from net.moce_ir import MoCEIR  # 导入模型主体 MoCEIR
from net.moe_ir import MoEIR  # 导入普通 MoE baseline
from net.msdr_moe_ir import MSDRMoEIR  # 导入推荐版 MSDR MoE
from net.msdr_stage_moe_ir import MSDRStageMoEIR  # 导入 stage-specific MSDR MoE

from options import train_options  # 导入训练参数解析函数
from utils.schedulers import LinearWarmupCosineAnnealingLR  # 导入预热+余弦退火学习率调度器
from data.dataset_utils import AIOTrainDataset, CDD11  # 导入训练数据集
from utils.loss_utils import FFTLoss  # 导入频域辅助损失

# Modified by zhangzhe.


def resolve_output_root(opt):
    """Resolve root folder for user-visible outputs.

    Keep backward compatibility:
    - default --output_path ("output/") -> use current working directory.
    - custom --output_path -> use that path as root.
    """
    output_path = getattr(opt, "output_path", None)
    if output_path is None:
        return os.getcwd()
    normalized = os.path.normpath(os.path.expanduser(output_path))
    if normalized in {"output", "."}:
        return os.getcwd()
    return normalized


DEGRADATION_ATTRS = ("noise", "haze", "rain", "blur", "low_light", "snow")


def degradation_to_attrs(name):
    name = str(name).lower()
    attrs = set()
    if name.startswith("denoise"):
        attrs.add("noise")
    if name in {"dehaze", "haze"} or "haze" in name.split("_"):
        attrs.add("haze")
    if name in {"derain", "rain"} or "rain" in name.split("_"):
        attrs.add("rain")
    if name == "deblur" or "blur" in name.split("_"):
        attrs.add("blur")
    if name in {"synllie", "lolv1", "low", "low_light"} or "low" in name.split("_"):
        attrs.add("low_light")
    if name == "snow" or "snow" in name.split("_"):
        attrs.add("snow")
    return attrs


def build_degradation_targets(de_id, opt, device):
    if isinstance(de_id, torch.Tensor):
        names = [opt.de_type[int(idx)] for idx in de_id.detach().cpu().view(-1)]
    elif isinstance(de_id, (list, tuple)):
        names = [str(item) for item in de_id]
    else:
        names = [str(de_id)]

    targets = torch.zeros(len(names), len(DEGRADATION_ATTRS), device=device)
    attr_to_idx = {name: idx for idx, name in enumerate(DEGRADATION_ATTRS)}
    for row, name in enumerate(names):
        for attr in degradation_to_attrs(name):
            targets[row, attr_to_idx[attr]] = 1.0
    return targets


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
        return MSDRMoEIR(**common_kwargs, num_degradation_attrs=len(DEGRADATION_ATTRS))
    if opt.model in {"MSDR_Stage_MoE_IR", "MSDR_Stage_MoE_IR_S"}:
        return MSDRStageMoEIR(**common_kwargs)
    return MoCEIR(
        **common_kwargs,
        with_complexity=opt.with_complexity,
        depth_type=opt.depth_type,
        rank_type=opt.rank_type,
        complexity_scale=opt.complexity_scale,
    )


class PLTrainModel(pl.LightningModule):  # 定义 Lightning 训练模块
    def __init__(self, opt):  # 构造函数，接收训练参数
        super().__init__()  # 调用父类初始化
        
        self.opt = opt  # 保存配置对象
        self.balance_loss_weight = opt.balance_loss_weight  # 保存平衡损失权重
        self.loss_type = opt.loss_type.lower()

        self.net = build_net(opt)  # 初始化 restoration 网络
        
             
        if self.loss_type == "fft":  # 若选择 fft 复合损失
            self.loss_fn = nn.L1Loss()  # 主损失为 L1 损失
            self.aux_fn = FFTLoss(loss_weight=self.opt.fft_loss_weight)  # 额外频域损失
        else:  # 否则
            self.loss_fn = nn.L1Loss()  # 仅使用 L1 损失
    
    def forward(self,x):  # 前向接口（推理时调用）
        return self.net(x)  # 调用底层网络
    
    def training_step(self, batch, batch_idx):  # 定义单步训练逻辑
        ([clean_name, de_id], degrad_patch, clean_patch) = batch  # 解包 batch：名称/退化类型、退化图块、干净图块
        restored = self.net(degrad_patch, de_id)  # 网络输出复原结果
        balance_loss = self.net.total_loss  # 读取网络内部平衡损失

        if self.loss_type == "fft":  # 若启用 fft 复合损失
            loss = self.loss_fn(restored,clean_patch)  # 计算像素域 L1 损失
            aux_loss = self.aux_fn(restored,clean_patch)  # 计算频域辅助损失
            loss += aux_loss  # 将辅助损失累加到总损失
        else:  # 若未启用 fft
            loss = self.loss_fn(restored,clean_patch)  # 仅计算 L1 损失

        degradation_logits = getattr(self.net, "degradation_logits", None)
        if degradation_logits is not None and self.opt.msdr_attr_loss_weight > 0:
            degradation_targets = build_degradation_targets(de_id, self.opt, degradation_logits.device)
            attr_loss = F.binary_cross_entropy_with_logits(degradation_logits, degradation_targets)
            loss += self.opt.msdr_attr_loss_weight * attr_loss
            self.log("MSDR_Attr", attr_loss, sync_dist=True)
            
        loss += self.balance_loss_weight * balance_loss  # 加入平衡损失项
        self.log("Train_Loss", loss, sync_dist=True)  # 记录训练总损失（多卡同步）
        self.log("Balance", balance_loss, sync_dist=True)  # 记录平衡损失（多卡同步）
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]  # 从优化器读取当前学习率
        self.log("LR Schedule", lr, sync_dist=True)  # 记录学习率（多卡同步）

        return loss  # 返回当前 step 的损失给 Lightning
        
    def lr_scheduler_step(self,scheduler,metric):  # 自定义调度器步进接口
        scheduler.step()  # 每个 epoch/step 调用一次调度器更新
    
    def configure_optimizers(self):  # 配置优化器与学习率调度器
        optimizer = optim.AdamW(self.parameters(), lr=self.opt.lr)  # 使用 AdamW，初始学习率由参数指定
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=self.opt.epochs)  # 默认调度策略
        
        if self.opt.fine_tune_from:  # 若为微调模式
            scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=1,max_epochs=self.opt.epochs)      # 微调时缩短预热并按总 epoch 设置
        return [optimizer],[scheduler]  # 按 Lightning 约定返回列表
                        


def main(opt):  # 训练主函数
    print("Options")  # 打印参数标题
    print(opt)  # 打印完整参数
    time_stamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')  # 生成当前时间戳字符串

    output_root = resolve_output_root(opt)  # 解析输出根目录（兼容旧默认行为）
    log_dir = os.path.join(output_root, "logs", time_stamp)  # 构造日志目录路径
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)  # 创建日志目录（不存在则创建）
    if opt.wblogger:  # 若启用 WandB 日志
        name = opt.model + "_" + time_stamp  # 组合实验名称
        logger  = WandbLogger(name=name, save_dir=log_dir, config=opt)  # 初始化 WandB 日志器
        
    else:  # 否则
        logger = TensorBoardLogger(save_dir=log_dir)  # 使用 TensorBoard 日志器

    # Create model  # 创建模型
    if opt.fine_tune_from:  # 若指定从已有 checkpoint 微调
        model = PLTrainModel.load_from_checkpoint(  # 从 checkpoint 加载模型权重
            os.path.join(opt.ckpt_dir, opt.fine_tune_from, "last.ckpt"), opt=opt)  # 指向目标目录中的 last.ckpt
    else:  # 否则
        model = PLTrainModel(opt)  # 从头初始化模型

    print(model)  # 打印模型结构
    checkpoint_path = os.path.join(opt.ckpt_dir, time_stamp)  # 构造当前实验的 checkpoint 输出目录
    pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)  # 创建 checkpoint 目录
    checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_path, every_n_epochs=5, save_top_k=-1, save_last=True)  # 每 5 个 epoch 保存一次并额外保存 last
    
    # Create datasets and dataloaders  # 创建数据集与数据加载器
    if "CDD11" in opt.trainset:  # 若训练集字符串包含 CDD11
        _, subset = opt.trainset.split("_")  # 按下划线解析子集名称
        trainset = CDD11(opt, split="train", subset=subset)  # 构建 CDD11 训练子集
    else:  # 否则
        trainset = AIOTrainDataset(opt)  # 构建通用训练数据集
        
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True, drop_last=True, num_workers=opt.num_workers)  # 构建训练 DataLoader
    
    # Create trainer  # 创建 Lightning Trainer
    trainer = pl.Trainer(max_epochs=opt.epochs,  # 训练总轮数
                         accelerator="gpu",  # 使用 GPU 加速
                         devices=opt.num_gpus,  # 使用 GPU 数量
                         strategy="ddp_find_unused_parameters_true",  # DDP 策略（允许未使用参数）
                         logger=logger,  # 配置日志器
                         callbacks=[checkpoint_callback],  # 注册 checkpoint 回调
                         accumulate_grad_batches=opt.accum_grad,  # 梯度累积步数
                         deterministic=True)  # 使用确定性模式
    
    # Optionally resume from a checkpoint  # 可选：从中断点继续训练
    if opt.resume_from:  # 若指定恢复训练目录
        checkpoint_path = os.path.join(opt.ckpt_dir, opt.resume_from, "last.ckpt")  # 读取该目录的 last.ckpt
    else:  # 否则
        checkpoint_path = None  # 不使用恢复点

    # Train model  # 开始训练
    trainer.fit(  # 调用 Lightning 训练入口
        model=model,  # 指定模型
        train_dataloaders=trainloader,  # 指定训练数据加载器
        ckpt_path=checkpoint_path  # 指定用于恢复的 checkpoint 路径
    )
    


if __name__ == '__main__':  # 脚本直接运行时执行
    train_opt = train_options()  # 解析训练参数
    main(train_opt)  # 启动训练主流程
