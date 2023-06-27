import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=256,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    opt = parser.parse_args()
     
    config = OmegaConf.load("configs/latent-diffusion/pororo_txt2img-1p4B-eval.yaml")  # TODO: Optionally download from same location as ckpt and chnage this logic
    #model = load_model_from_config(config, "logs/2022-12-08T13-41-28_pororo_txt2img-1p4B-train/checkpoints/epoch=000087.ckpt")  # TODO: check path
    model = load_model_from_config(config, "/ubc/cs/research/shield/projects/trahman8/snap_research/latent-diffusion_text2Image_original/logs/2022-12-05T13-18-34_pororo_txt2img-1p4B-train/checkpoints/epoch=000111.ckpt")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    prompt = opt.prompt


    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    num_vid = 4
   
    with torch.no_grad():
        with model.ema_scope():
            uc = None
            if opt.scale != 1.0:
                uc = torch.zeros(1, num_vid, 77, 1280).to(device)
                for j in range(num_vid):
                    uc[:,j] = model.get_learned_conditioning(opt.n_samples * [""])

            for n in trange(opt.n_iter, desc="Sampling"):
                all_samples=list()
                c = torch.zeros(1, num_vid, 77, 1280).to(device)
                batch_clip_video = prompt.split(';')
                for j in range(len(batch_clip_video)):
                    c[:, j] = model.get_learned_conditioning(opt.n_samples * [batch_clip_video[j]])

                #c = model.get_learned_conditioning(opt.n_samples * [prompt])
                shape = [4, num_vid, opt.H//8, opt.W//8]
    
                samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                 conditioning=c,
                                                 batch_size=opt.n_samples,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,
                                                 eta=opt.ddim_eta)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

                for x_sample in x_samples_ddim:
                    if x_sample.dim() > 3:
                        for l, x_sample_frame in enumerate(x_sample):
                            x_sample_frame = 255. * rearrange(x_sample_frame.cpu().numpy(), 'c h w -> h w c')
                            Image.fromarray(x_sample_frame.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:04}_{l:02}.png"))
                            base_count += 1
                    continue
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:04}.png"))
                    base_count += 1
                all_samples.append(x_samples_ddim)
                grid = torch.stack(all_samples, 0)
                p = 1
                grid = rearrange(grid, 'p b t c h w -> (p b) t c h w')
                if grid.dim() > 4:
                    for i in range(grid.shape[0]):
                        grid_ = make_grid(grid[i], nrow=12)#opt.n_samples)
                        grid_ = 255. * rearrange(grid_, 'c h w -> h w c').cpu().numpy()
                        Image.fromarray(grid_.astype(np.uint8)).save(os.path.join(outpath, f'test'+str(n)+'.png'))


    # additionally, save as grid
    '''grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b t c h w -> (n b) t c h w')
    if grid.dim() > 4:
        for i in range(grid.shape[0]):
            grid_ = make_grid(grid[i], nrow=12)#opt.n_samples)
            # to image
            grid_ = 255. * rearrange(grid_, 'c h w -> h w c').cpu().numpy()
            #Image.fromarray(grid_.astype(np.uint8)).save(os.path.join(outpath, f'{prompt.replace(" ", "-")}_{i:01}.png'))
            Image.fromarray(grid_.astype(np.uint8)).save(os.path.join(outpath, f'test.png'))

            print(f"Your samples are ready and waiting four you here: \n{outpath} \nEnjoy.")'''
