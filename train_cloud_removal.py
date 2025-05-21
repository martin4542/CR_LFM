import os
import shutil
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial

import torch
import torchvision
import torch.nn.functional as F
# from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint as odeint

from EMA import EMA
from models.DiT_SEN12 import DiT
from diffusers.models import AutoencoderKL
from datasets_prep.SEN12_dataset import SEN12Dataset


# faster training
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
assert torch.cuda.is_available(), "CUDA is not available."

DTYPE = torch.float32
HIDDEN_FEATURE_DIM = 1024
LATENT_SIZE = 32



def parse_args():
    parser = argparse.ArgumentParser("flow-matching parameters")
    parser.add_argument("--seed", type=int, default=1024, help="Seed used for initialization")

    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--model_ckpt", type=str, default=None, help="Model ckpt to init from")

    # model parameters
    parser.add_argument("--image_size", type=int, default=256, help="size of image")
    parser.add_argument("--depth", type=int, default=12, help="num of DiT layers")
    parser.add_argument("--patch_size", type=int, default=4, help="num of patches")
    parser.add_argument("--num_heads", type=int, default=16, help="num of attention heads")
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsample rate of input image by the autoencoder",
    )
    parser.add_argument("--scale_factor", type=float, default=0.18215, help="size of image")
    parser.add_argument("--num_in_channels", type=int, default=4, help="in channel image")
    parser.add_argument("--num_out_channels", type=int, default=4, help="in channel image")
    parser.add_argument("--nf", type=int, default=256, help="channel of model")
    parser.add_argument(
        "--num_res_blocks",
        type=int,
        default=2,
        help="number of resnet blocks per scale",
    )
    parser.add_argument(
        "--attn_resolutions",
        nargs="+",
        type=int,
        default=(16,),
        help="resolution of applying attention",
    )
    parser.add_argument(
        "--ch_mult",
        nargs="+",
        type=int,
        default=(1, 1, 2, 2, 4, 4),
        help="channel mult",
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="drop-out rate")
    parser.add_argument("--label_dim", type=int, default=0, help="label dimension, 0 if unconditional")
    parser.add_argument(
        "--augment_dim",
        type=int,
        default=0,
        help="dimension of augmented label, 0 if not used",
    )
    parser.add_argument(
        "--label_dropout",
        type=float,
        default=0.0,
        help="Dropout probability of class labels for classifier-free guidance",
    )

    # Original ADM
    parser.add_argument("--layout", action="store_true")
    parser.add_argument("--use_origin_adm", action="store_true")
    parser.add_argument("--use_scale_shift_norm", type=bool, default=True)
    parser.add_argument("--resblock_updown", type=bool, default=False)
    parser.add_argument("--use_new_attention_order", type=bool, default=False)
    parser.add_argument("--centered", action="store_false", default=True, help="-1,1 scale")
    parser.add_argument("--resamp_with_conv", type=bool, default=True)
    parser.add_argument("--num_head_upsample", type=int, default=-1, help="number of head upsample")
    parser.add_argument("--num_head_channels", type=int, default=-1, help="number of head channels")

    parser.add_argument("--pretrained_autoencoder_ckpt", type=str, default="stabilityai/sd-vae-ft-mse")

    # training
    parser.add_argument("--exp", default="experiment_cifar_default", help="name of experiment")
    parser.add_argument("--result_path", default="./results", help="path to save results")
    parser.add_argument("--dataset", default="cifar10", help="name of dataset")
    parser.add_argument("--datadir", default="/workspace/generative_model/data")
    parser.add_argument("--num_timesteps", type=int, default=200)
    parser.add_argument(
        "--use_grad_checkpointing",
        action="store_true",
        default=False,
        help="Enable gradient checkpointing for mem saving",
    )

    parser.add_argument("--batch_size", type=int, default=128, help="input batch size")
    parser.add_argument("--num_epoch", type=int, default=1200)

    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate g")

    parser.add_argument("--beta1", type=float, default=0.5, help="beta1 for adam")
    parser.add_argument("--beta2", type=float, default=0.9, help="beta2 for adam")
    parser.add_argument("--no_lr_decay", action="store_true", default=False)

    parser.add_argument("--use_ema", action="store_true", default=False, help="use EMA or not")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="decay rate for EMA")

    parser.add_argument("--save_content", action="store_true", default=False)
    parser.add_argument(
        "--save_content_every",
        type=int,
        default=10,
        help="save content for resuming every x epochs",
    )
    parser.add_argument("--save_ckpt_every", type=int, default=25, help="save ckpt every x epochs")
    parser.add_argument("--plot_every", type=int, default=5, help="plot every x epochs")

    return parser.parse_args()


def load_model_from_ckpt(model, optimizer, scheduler, target_path, device):
    checkpoint_file = os.path.join(target_path, "content.pth")
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_dict"])
    init_epoch = checkpoint["epoch"]
    epoch = init_epoch
    # load G
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    global_step = checkpoint["global_step"]

    print(f"=> Resume checkpoitn from epoch: {checkpoint['epoch']}")
    del checkpoint

    return model, optimizer, scheduler, epoch, global_step


def load_model_from_usr_ckpt(model, target_path, ckpt_name, device):
    checkpoint_file = os.path.join(target_path, ckpt_name)
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["model_dict"])
    init_epoch = int(ckpt_name.split("_")[-1][:-4])
    epoch = init_epoch
    global_step = 0

    print(f"=> Resume checkpoitn from epoch: {checkpoint['epoch']}")
    del checkpoint

    return model, epoch, global_step


def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def get_weight(model):
    size_all_mb = sum(p.numel() for p in model.parameters()) / 1024**2
    return size_all_mb


def sample_from_model(model, x_0):
    t = torch.tensor([1.0, 0.0], dtype=x_0.dtype, device="cuda")
    fake_image = odeint(model, x_0, t, atol=1e-5, rtol=1e-5, adjoint_params=model.func.parameters())
    # fake_image = odeint(model, x_0, t, atol=1e-5, rtol=1e-5)
    return fake_image


def save_result_image(fake_image, target_path):
    if not os.path.exists(os.path.dirname(target_path)):
        os.makedirs(os.path.dirname(target_path))
    torchvision.utils.save_image(
        fake_image,
        os.path.join(target_path),
        normalize=True,
        value_range=(-1, 1)
    )


def train(args):
    # Set up accelerator
    device = "cuda"

    # Load dataset
    dataset = SEN12Dataset(
        args.datadir,
        mode="train",
        random_flip=True,
        image_size=args.image_size
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Load validation dataset
    val_dataset = SEN12Dataset(
        args.datadir,
        mode="val",
        image_size=args.image_size
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # load model
    model = DiT(
        depth           = args.depth,
        hidden_size     = HIDDEN_FEATURE_DIM,
        patch_size      = args.patch_size,
        num_heads       = args.num_heads,
        img_resolution  = LATENT_SIZE, # latent feature size: 32x32
        in_channels     = args.num_in_channels
    )
    if args.use_grad_checkpointing:
        model.set_gradient_checkpointing()

    # define latent feature extractor
    latent_feature_extractor = AutoencoderKL.from_pretrained(args.pretrained_autoencoder_ckpt).to(device, dtype=DTYPE)
    latent_feature_extractor = latent_feature_extractor.eval()
    latent_feature_extractor.train = False
    for param in latent_feature_extractor.parameters():
        param.requires_grad = False

    # models to GPU & set DataParallel
    model = torch.nn.DataParallel(model)
    model = model.to(device, dtype=DTYPE)
    latent_feature_extractor = latent_feature_extractor.to(device, dtype=DTYPE)

    # define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    if args.use_ema:
        optimizer = EMA(optimizer, ema_decay=args.ema_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epoch, eta_min=1e-5)

    # define experiment result save directory
    target_path = os.path.join(args.result_path, args.exp)
    if not os.path.exists(target_path): os.makedirs(target_path)

    # if resume training from checkpoint
    if args.resume or os.path.exists(os.path.join(target_path, "content.pth")):
        model, optimizer, scheduler, epoch, global_step = load_model_from_ckpt(model, optimizer, scheduler, target_path, device)

    elif args.model_ckpt and os.path.exists(os.path.join(target_path, args.model_ckpt)):
        model, epoch, global_step = load_model_from_usr_ckpt(model, target_path, args.model_ckpt, device)

    else:
        init_epoch, epoch, global_step = 0, 0, 0

    # start training
    for epoch in range(init_epoch, args.num_epoch):
        avg_loss = 0

        for iteration, (src_image, target_image, ridar_image) in enumerate(tqdm(data_loader)):
            src_image = src_image.to(device, dtype=DTYPE)
            target_image = target_image.to(device, dtype=DTYPE)
            ridar_image = ridar_image.to(device, dtype=DTYPE)

            # src: cloudy image (z_1, random noise in ordinary flow matching)
            # target: cloud-free image (z_0, original image in ordinary flow matching)
            src_latent = latent_feature_extractor.encode(src_image).latent_dist.sample().mul_(args.scale_factor)
            target_latent = latent_feature_extractor.encode(target_image).latent_dist.sample().mul_(args.scale_factor)

            # timestep t
            t = torch.rand((src_latent.size(0)), dtype=DTYPE, device=device)
            t = t.view(-1, 1, 1, 1)

            # get intermediate latent feature at timestep t
            intermediate_latent = (1 - t) * target_latent + (1e-5 + (1 - 1e-5) * t) * src_latent
            u = (1 - 1e-5) * src_latent - target_latent

            # estimate velocity & compute loss
            v = model(t.squeeze(), intermediate_latent, ridar_image)
            loss = F.mse_loss(u, v)
            # loss = F.smooth_l1_loss(u, v)

            avg_loss += loss.item()
            # backprop loss
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            global_step += 1
            if (iteration % 100) == 0:
                print(f"{epoch}/{args.num_epoch}|{iteration}_Loss: {avg_loss / 100}")
                avg_loss = 0

        # step scheduler
        if not args.no_lr_decay:
            scheduler.step()

        # inference 
        # remove dataparallel (occurs error on odeint)
        validation_model = model.module.to(device).eval()
        if (epoch % args.plot_every) == 0:
            with torch.no_grad():
                avg_mae_loss = 0
                for idx, (src_image, target_image, ridar_image) in enumerate(tqdm(val_loader)):
                    src_image = src_image.to(device, dtype=DTYPE)
                    target_image = target_image.to(device, dtype=DTYPE)
                    ridar_image = ridar_image.to(device, dtype=DTYPE)

                    src_latent = latent_feature_extractor.encode(src_image).latent_dist.sample().mul_(args.scale_factor)
                    sample_model = partial(validation_model, y=ridar_image)
                    fake_sample = sample_from_model(sample_model, src_latent)[-1]
                    fake_image = latent_feature_extractor.decode(fake_sample / args.scale_factor).sample

                    mae_loss = F.l1_loss(target_image, fake_image)
                    avg_mae_loss += mae_loss.item()

                    if idx % 2 == 0: 
                        result_img_path = os.path.join(target_path, str(epoch), f"{idx}.png")
                        save_result_image(fake_image, result_img_path)

                    if idx >= 10: break

                print(f"Average MAE: {avg_mae_loss / idx}")
                print("Finish Validation")

            if (args.save_content) and (epoch % args.save_content_every == 0):
                    print("Saving content.")
                    content = {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "args": args,
                        "model_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict()
                    }

                    torch.save(content, os.path.join(target_path, "content.pth"))


if __name__ == "__main__":
    args = parse_args()

    train(args)