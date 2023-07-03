import os
import copy
import fire
import torch
import random
import logging
import numpy as np
from PIL import Image
from unet import UNetModel
from torchvision.utils import make_grid
from torchvision import datasets, transforms

def main(
    data_folder: str = "~/data", output_folder: str = './cifar10-cm',                               # Data & IO
    batch_size: int = 64, device: str = 'cuda', seed: int = 0,                                      # Training basic
    img_size: int = 32, img_channels : int = 3,                                                     # Image shape
    iter_log: int = 100, iter_fig: int = 500, iter_save : int = 10000, iter_train : int = 800000,   # Time stage
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
        ), batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True,
    )

    ### Define model and optimizer
    model = UNetModel(image_size=img_size, in_channels=img_channels, model_channels=128, out_channels=img_channels, num_res_blocks=2, attention_resolutions=[32,16,8], num_heads=4).to(device)
    model_ema = UNetModel(image_size=img_size, in_channels=img_channels, model_channels=128, out_channels=img_channels, num_res_blocks=2, attention_resolutions=[32,16,8], num_heads=4).to(device)
    model_ema.load_state_dict(copy.deepcopy(model.state_dict()))
    optim = torch.optim.RAdam(model.parameters(), lr=1e-4)

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
    def forward(model, x_t, sigmas, sigma_data=0.5):
        c_skip = sigma_data**2 / ((sigmas - sigma_min) ** 2 + sigma_data**2)
        c_out = ((sigmas - sigma_min) * sigma_data / (sigmas**2 + sigma_data**2) ** 0.5)
        c_in = 1 / (sigmas**2 + sigma_data**2) ** 0.5
        rescaled_t = 1000 * 0.25 * torch.log(sigmas + 1e-44)            # ???
        model_output = model(c_in[..., None, None, None] * x_t, rescaled_t)
        denoised = c_out[..., None, None, None] * model_output + c_skip[..., None, None, None] * x_t
        return denoised
        
    ### Training loop
    curr_iter = 0
    while curr_iter < iter_train:
        for x, _ in dataloader:
            x = x.to(device)

            # Get EMA decay rate and maximum step by schedule function
            ema, num_scales = ema_and_scales_fn(curr_iter)

            # Random select a time step from current maximum step section to train
            indices = torch.randint(0, num_scales - 1, (x.shape[0],), device=device)

            # Compute time boundary as EMA paper setting (Consistency Model paper page 4 bottom)
            t1 = (sigma_max ** (1 / rho) + indices / (num_scales - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
            t2 = (sigma_max ** (1 / rho) + (indices + 1) / (num_scales - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

            # Construct t_(n+1) noisy input
            x_t1 = x + torch.randn_like(x) * t1[..., None, None, None]

            # Denoise with online network
            distiller = forward(model, x_t1, t1, sigma_data=sigma_data)

            # Construct t_n noisy input via Euler solver (Might can construct directly)
            x_t2 = (x_t1 + (x_t1 - x) / t1[..., None, None, None] * (t2 - t1)[..., None, None, None]).detach()

            # Denoise with target network
            distiller_target = forward(model_ema, x_t2, t2, sigma_data=sigma_data).detach()

            # Compute weighted loss with consistency loss (weight definition is described in EDM paper page 3 Table 1)
            weightings = 1.0 / t1**2 + 1.0 / sigma_data**2
            diffs = (distiller - distiller_target) ** 2
            loss = (torch.mean(diffs, dim=(1, 2, 3)) * weightings).mean()

            # Update online network
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Update target network manually and accumulate global step (notation is k in Consistency Model paper Algorithm 3)
            with torch.no_grad():
                for targ, src in zip(model_ema.parameters(), model.parameters()):
                    targ.detach().mul_(ema).add_(src, alpha=1 - ema)

            ### Log & save
            if curr_iter % iter_log == 0:
                logger.info("[{:>6} / {:>6}]  Loss: {:.6f}".format(curr_iter, iter_train, loss.item()))
            if curr_iter % iter_save == 0:
                torch.save(model_ema.state_dict(), os.path.join(output_folder, '{}.pth'.format(str(curr_iter).zfill(6))))
            if curr_iter % iter_fig == 0:
                with torch.no_grad():
                    out = forward(model_ema, torch.randn_like(x).to(device) * sigma_max, torch.ones(x.shape[0]).to(device) * sigma_max).clamp(-1, 1)
                    out = make_grid(out)
                    out = (out.permute(1, 2, 0) * 127.5 + 127.5).cpu().numpy().astype(np.uint8)
                    Image.fromarray(out).save(os.path.join(output_folder, '{}.jpg'.format(str(curr_iter).zfill(6))))
            if curr_iter > iter_train + 1:
                break
            curr_iter += 1

if __name__ == '__main__':
    fire.Fire(main)