import torch
from matplotlib import pyplot as plt

from models import UNet, Diffusion


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.savefig('result.jpg')
    plt.show()


def main():
    device = "cuda"
    model = UNet().to(device)
    ckpt = torch.load("weights/ddpm_ckpt.pt")
    model.load_state_dict(ckpt)
    diffusion = Diffusion(img_size=64, device=device)
    x = diffusion.sample(model, n=1)
    plot_images(x)


if __name__ == '__main__':
    main()