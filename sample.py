import os
import fire
import torch
import random
import numpy as np
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from unet import UNetModel
from torchvision.utils import save_image

@torch.no_grad()
def main(
    model_path : str = './cifar10-cm/800000.pth', output_folder : str = './cifar10-cm/sample',      # Data & IO
    batch_size : int = 64, device : str = 'cuda', num_sample : int = 64, seed : int = 0,            # Inference basic
    img_size : int = 32, img_channels : int = 3,                                                    # Image shape
    grid_form : bool = True,                                                                        # Save format
    sigma_max: float = 80.0, sigma_min: float = 0.002,                                              # EDM hyper-parameters
    ):
    os.makedirs(output_folder, exist_ok=True)

    ### For reproduce result
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = UNetModel(image_size=img_size, in_channels=img_channels, model_channels=128, out_channels=img_channels, num_res_blocks=2, attention_resolutions=[32,16,8], num_heads=4).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

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

    ### Sampling loop
    ctr = 0
    bar = tqdm(total=num_sample)
    x_shape=[batch_size, img_channels, img_size, img_size]
    while ctr < num_sample:
        with torch.no_grad():
            imgs = torch.randn(x_shape, device=device, dtype=torch.float)
            imgs = forward(model, imgs * sigma_max, torch.ones(imgs.shape[0]).to(device) * sigma_max).clamp(-1, 1)
            imgs = (imgs / 2 + 0.5)

        if grid_form:
            save_image(imgs, os.path.join(output_folder, '{}.png'.format(str(ctr).zfill(6))))
            ctr += imgs.shape[0]
            bar.update(n=imgs.shape[0])
        else:
            imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
            for img in imgs:
                img = Image.fromarray((img * 255).astype(np.uint8))
                img.save(os.path.join(output_folder, '{}.png'.format(str(ctr).zfill(6))))
                ctr += 1
                bar.update(n=1)

if __name__ == '__main__':
    fire.Fire(main)