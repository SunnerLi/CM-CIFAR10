import os
import copy
import fire
import torch
import random
import logging
import numpy as np
from unet import UNetModel
from torchvision.utils import save_image
from torchvision import datasets, transforms

def main(
    data_folder: str = "~/data", output_folder: str = './cifar10-cm',                               # IO
    device: str = 'cuda', seed: int = 0, lr: float = 5e-5,                                          # Training basic
    batch_size: int = 64, accumulate_grad_batches: int = 4,                                         # Batch size
    data_size: int = 32, data_channels : int = 3,                                                   # Data shape
    model_channels: int = 128, use_scale_shift_norm: bool = True, resblock_updown: bool = True,     # Model setting
    iter_log: int = 50, iter_fig: int = 200, iter_save : int = 10000, iter_train : int = 80000,     # Time stage
    start_scales: int = 2, end_scales: int = 150, start_ema: float = 0.95,                          # EDM hyper-parameters
    sigma_max: float = 80.0, sigma_min: float = 0.002, rho: float = 7.0, sigma_data: float = 0.5,   # EDM hyper-parameters
):
    ### For reproduce result
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ### Output stuff
    os.makedirs(output_folder, exist_ok=True)
    logging.basicConfig(filename=os.path.join(output_folder, 'log.txt'), level=logging.DEBUG)
    logger = logging.getLogger()

    ### Loading dataset
    dataloader = torch.utils.data.DataLoader(
        dataset=datasets.CIFAR10(
            root=data_folder, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        ), batch_size=batch_size*accumulate_grad_batches, shuffle=True, num_workers=4, drop_last=True,
    )

    ### Define model and optimizer
    model = UNetModel(image_size=data_size, in_channels=data_channels, model_channels=model_channels, out_channels=data_channels, num_res_blocks=2, attention_resolutions=[32,16,8], num_heads=4, use_scale_shift_norm=use_scale_shift_norm, resblock_updown=resblock_updown).to(device)
    model_ema = copy.deepcopy(model)
    optim = torch.optim.RAdam(model.parameters(), lr=lr)

    ### Obtain schedule function for EMA decay rate schedule \mu(.) & step schedule N(.)
    ### Formula can be referred from Consistency Models paper Appendix C. page 26. 
    def ema_and_scales_fn(step):
        scales = np.ceil(np.sqrt((step / iter_train) * ((end_scales + 1) ** 2 - start_scales**2) + start_scales**2) - 1).astype(np.int32)
        scales = np.maximum(scales, 1)
        c = -np.log(start_ema) * start_scales
        target_ema = np.exp(-c / scales)
        scales = scales + 1
        return target_ema, scales

    ### Define wrapper forward function to perform meet boundary condition
    ### Formula of differentiable functions can be referred from Consistency Models paper Appendix C. page 25~26.
    ### Definition of rescaled_t is noise condition which is defined in EDM paper Table 1. page 3. 
    def forward(model, x_t, sigmas, sigma_data=0.5, y=None):
        c_skip = sigma_data**2 / ((sigmas - sigma_min) ** 2 + sigma_data**2)
        c_out = ((sigmas - sigma_min) * sigma_data / (sigmas**2 + sigma_data**2) ** 0.5)
        c_in = 1 / (sigmas**2 + sigma_data**2) ** 0.5
        rescaled_t = 1000 * 0.25 * torch.log(sigmas + 1e-44)
        model_output = model(c_in[..., None, None, None] * x_t, rescaled_t, y=y)
        denoised = c_out[..., None, None, None] * model_output + c_skip[..., None, None, None] * x_t
        return denoised
        
    ### Training loop (Please manually set y as two forward functions parameter if you want to perform conditional generation)
    curr_iter = 0
    optim.zero_grad()
    sample_z = None
    while curr_iter < iter_train and curr_iter < 4000:
        for X, Y in dataloader:
            losses = [0.0]
            for mini_batch_idx in range(accumulate_grad_batches):
                x = X[mini_batch_idx*batch_size : (mini_batch_idx+1)*batch_size].to(device)
                y = Y[mini_batch_idx*batch_size : (mini_batch_idx+1)*batch_size].to(device)

                # Get EMA decay rate and maximum step by schedule function
                ema, num_scales = ema_and_scales_fn(curr_iter)

                # Random select a time step from current maximum step section to train
                indices = torch.randint(0, num_scales - 1, (x.shape[0],), device=device)

                # Compute time boundary as EMA paper setting (Consistency Model paper page 4 bottom)
                t1 = (sigma_max ** (1 / rho) + indices / (num_scales - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
                t2 = (sigma_max ** (1 / rho) + (indices + 1) / (num_scales - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

                # Construct t_(n+1) noisy input and denoise with online network
                x_t1 = x + torch.randn_like(x) * t1[..., None, None, None]
                distiller_online = forward(model, x_t1, t1, sigma_data=sigma_data, y=None)

                # Construct t_n noisy input via Euler solver (Might can construct directly) and denoise with target network
                x_t2 = (x_t1 + (x_t1 - x) / t1[..., None, None, None] * (t2 - t1)[..., None, None, None]).detach()
                distiller_target = forward(model_ema, x_t2, t2, sigma_data=sigma_data, y=None).detach()

                # Compute weighted loss with consistency loss (weight definition is described in EDM paper page 3 Table 1)
                weightings = 1.0 / t1**2 + 1.0 / sigma_data**2
                diffs = (distiller_online - distiller_target) ** 2
                loss = (torch.mean(diffs, dim=(1, 2, 3)) * weightings).mean()
                loss.backward()
                losses.append(loss.item())

            # Update online network
            optim.step()
            optim.zero_grad()

            # Update target network manually 
            with torch.no_grad():
                for targ, src in zip(model_ema.parameters(), model.parameters()):
                    targ.detach().mul_(ema).add_(src, alpha=1 - ema)

            ### Log & save
            if curr_iter % iter_log == 0:
                logger.info("[{:>6} / {:>6}]  Loss: {:.6f}".format(curr_iter, iter_train, np.mean(losses)))
            if curr_iter % iter_save == 0:
                torch.save(model_ema.state_dict(), os.path.join(output_folder, '{}.pth'.format(str(curr_iter).zfill(6))))
            if curr_iter % iter_fig == 0:
                with torch.no_grad():
                    sample_z = torch.randn_like(x).to(device) if sample_z is None else sample_z
                    out = forward(model_ema, sample_z * sigma_max, torch.ones(x.shape[0]).to(device) * sigma_max).clamp(-1, 1)
                    save_image(out / 2 + 0.5, os.path.join(output_folder, '{}.jpg'.format(str(curr_iter).zfill(6))))
            if curr_iter > iter_train + 1:
                break

            # Accumulate global step (notation is k in Consistency Model paper Algorithm 3)
            curr_iter += 1
    
    # Save final trained result
    torch.save(model_ema.state_dict(), os.path.join(output_folder, 'last.pth'))

if __name__ == '__main__':
    fire.Fire(main)