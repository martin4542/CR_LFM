import os
import argparse
from tqdm import tqdm
from functools import partial

import torch
import torchvision
import torch.nn.functional as F
from diffusers.models import AutoencoderKL
from torchdiffeq import odeint_adjoint as odeint

from models.DiT_SEN12 import DiT
from datasets_prep.SEN12_dataset import SEN12Dataset


DTYPE = torch.float32
HIDDEN_FEATURE_DIM = 1024
LATENT_SIZE = 32



def parse_args():
    parser = argparse.ArgumentParser("flow-matching parameters")
    parser.add_argument(
        "--generator",
        type=str,
        default="determ",
        help="type of seed generator",
        choices=["dummy", "determ", "determ-indiv"],
    )
    parser.add_argument("--seed", type=int, default=42, help="seed used for initialization")
    parser.add_argument("--compute_fid", action="store_true", default=False, help="whether or not compute FID")
    parser.add_argument("--compute_nfe", action="store_true", default=False, help="whether or not compute NFE")
    parser.add_argument("--measure_time", action="store_true", default=False, help="wheter or not measure time")
    parser.add_argument("--epoch_id", type=int, default=1000)
    parser.add_argument("--n_sample", type=int, default=50000, help="number of sampled images")

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
    parser.add_argument("--nf", type=int, default=256, help="channel of image")
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
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="Scale for classifier-free guidance")

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
    parser.add_argument("--output_log", type=str, default="")

    #######################################
    parser.add_argument("--ckpt_path", type=str, default="/workspace/generative_model/CR_LFM/results/SEN12_0515/content.pth")
    parser.add_argument("--datadir", default="/workspace/generative_model/data")
    parser.add_argument(
        "--real_img_dir",
        default="./pytorch_fid/cifar10_train_stat.npy",
        help="directory to real images for FID computation",
    )
    parser.add_argument("--num_steps", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32, help="sample generating batch size")

    # sampling argument
    parser.add_argument("--exp", default="SEN12_0515", help="name of experiment")
    parser.add_argument("--result_path", default="./results", help="path to save results")
    parser.add_argument("--use_karras_samplers", action="store_true", default=False)
    parser.add_argument("--atol", type=float, default=1e-5, help="absolute tolerance error")
    parser.add_argument("--rtol", type=float, default=1e-5, help="absolute tolerance error")
    parser.add_argument(
        "--method",
        type=str,
        default="dopri5",
        help="solver_method",
        choices=[
            "dopri5",
            "dopri8",
            "adaptive_heun",
            "bosh3",
            "euler",
            "midpoint",
            "rk4",
            "heun",
            "multistep",
            "stochastic",
            "dpm",
        ],
    )
    parser.add_argument("--step_size", type=float, default=0.01, help="step_size")
    parser.add_argument("--perturb", action="store_true", default=False)

    # ddp
    parser.add_argument("--num_proc_node", type=int, default=1, help="The number of nodes in multi node env.")
    parser.add_argument("--num_process_per_node", type=int, default=1, help="number of gpus")
    parser.add_argument("--node_rank", type=int, default=0, help="The index of node.")
    parser.add_argument("--local_rank", type=int, default=0, help="rank of process in the node")
    parser.add_argument("--master_address", type=str, default="127.0.0.1", help="address for master")
    parser.add_argument("--master_port", type=str, default="6000", help="port for master")
    return parser.parse_args()


def load_model_from_ckpt(checkpoint_path, model, device):
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint["model_dict"])
    del checkpoint
    return model.eval()


# def sample_from_model(args, model, x_0):
#     t = torch.tensor([1.0, 0.0], dtype=x_0.dtype, device="cuda")
#     # fake_image = odeint(model, x_0, t, atol=1e-5, rtol=1e-5, adjoint_params=model.func.parameters())
#     # fake_image = odeint(model, x_0, t, atol=1e-5, rtol=1e-5)
#     fake_image = odeint(
#         model,
#         x_0,
#         t
#     )
#     return fake_image

def sample_from_model(args, model, x_0, model_kwargs):
    options = {"dtype": torch.float64}
    t = torch.tensor([1.0, 0.0], device="cuda")

    def denoiser(t, x_0):
        return model(t, x_0, **model_kwargs)

    fake_image = odeint(
        denoiser,
        x_0,
        t,
        method=args.method,
        atol=args.atol,
        rtol=args.rtol,
        adjoint_method=args.method,
        adjoint_atol=args.atol,
        adjoint_rtol=args.rtol,
        options=options,
        adjoint_params=model.parameters()
    )
    return fake_image


def save_result_image(fake_image, target_path):
    if not os.path.exists(os.path.dirname(target_path)):
        os.makedirs(os.path.dirname(target_path))
    torchvision.utils.save_image(
        fake_image,
        os.path.join(target_path),
        normalize=False
    )


def test(args):
    device = "cuda"

    torch.set_grad_enabled(False)

    # Load Test dataset
    test_dataset = SEN12Dataset(
        args.datadir,
        mode="test",
        random_flip=False,
        image_size=args.image_size
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Load Model & Load checkpoint
    model = DiT(
        depth           = args.depth,
        hidden_size     = HIDDEN_FEATURE_DIM,
        patch_size      = args.patch_size,
        num_heads       = args.num_heads,
        img_resolution  = LATENT_SIZE, # latent feature size: 32x32
        in_channels     = args.num_in_channels
    )
    model = torch.nn.DataParallel(model)
    model = load_model_from_ckpt(args.ckpt_path, model, device)
    # model = model.to(device, dtype=DTYPE)
    model = model.module.to(device, dtype=DTYPE).eval()

    # define latent feature extractor
    latent_feature_extractor = AutoencoderKL.from_pretrained(args.pretrained_autoencoder_ckpt).to(device, dtype=DTYPE)
    latent_feature_extractor = latent_feature_extractor.eval()
    latent_feature_extractor.train = False
    for param in latent_feature_extractor.parameters():
        param.requires_grad = False

    # define experiment result save directory
    target_path = os.path.join(args.result_path, args.exp, "test_results")
    if not os.path.exists(target_path): os.makedirs(target_path)

    with torch.no_grad():
        avg_mae_loss = 0
        for idx, (src_image, target_image, ridar_image) in enumerate(tqdm(test_loader)):
            src_image = src_image.to(device, dtype=DTYPE)
            target_image = target_image.to(device, dtype=DTYPE)
            ridar_image = ridar_image.to(device, dtype=DTYPE)

            src_latent = latent_feature_extractor.encode(src_image).latent_dist.sample().mul_(args.scale_factor)
            # sample_model = partial(model, y=ridar_image)
            # fake_sample = sample_from_model(args, sample_model, src_latent)[-1]
            model_kwargs = dict(y=ridar_image)
            fake_sample = sample_from_model(args, model, src_latent, model_kwargs)[-1]
            fake_image = latent_feature_extractor.decode(fake_sample / args.scale_factor).sample

            mae_loss = F.l1_loss(target_image, fake_image)
            avg_mae_loss += mae_loss.item()
            print(mae_loss.item())

            result_image_path = os.path.join(target_path, f"{idx}.png")
            save_result_image(fake_image, result_image_path)
            target_image_path = os.path.join(target_path, f"{idx}_gt.png")
            save_result_image(target_image, target_image_path)


if __name__ == "__main__":
    args = parse_args()

    test(args)
