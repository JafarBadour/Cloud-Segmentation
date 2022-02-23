from pathlib import Path
import numpy as np
from tifffile import imsave, imread
from unet import UNet
import torch


def numpy2tensor(x):
    x = x.astype(float)
    x = torch.from_numpy(x)
    x = x.type(torch.FloatTensor).cuda()
    return x


def run_inference():
    
    head = "/" if True else "/home/jafar/Desktop/On Cloud N/"
    ROOT_DIRECTORY = Path(f"{head}codeexecution")
    PREDICTIONS_DIRECTORY = ROOT_DIRECTORY / "predictions"
    INPUT_IMAGES_DIRECTORY = ROOT_DIRECTORY / "data/test_features"

    PATH = "./model_checkpts/true-image-unet-loss-epoch-9-loss-0.12250382370892532.pth"
    model = UNet(4, 1)

    model.load_state_dict(torch.load(PATH))
    model = model.cuda()

    model.eval()

    BANDS = ["B02", "B03", "B04", "B08"]

    chip_ids = (
        pth.name for pth in INPUT_IMAGES_DIRECTORY.iterdir() if not pth.name.startswith(".")
    )

    for chip_id in chip_ids:
        band_arrs = []
        for band in BANDS:
            band_arr = imread(INPUT_IMAGES_DIRECTORY / f"{chip_id}/{band}.tif")
            # band_arr = (1.0 * band_arr - band_arr.min())
            # band_arr = 255.0* band_arr/band_arr.max()
            band_arrs.append(band_arr)
        chip_arr = np.stack(band_arrs)
        # could do something useful here with chip_arr ;-)
        chip_arr = chip_arr.reshape(1, 4, 512, 512)

        x_input = numpy2tensor(chip_arr)
        prediction = model(x_input)
        prediction = prediction.cpu().detach().numpy()
        prediction = prediction.reshape(512, 512)
        s  = prediction.max() - prediction.min()
        if s != 0:
            prediction = (prediction - prediction.min()) / s
        output_path = PREDICTIONS_DIRECTORY / f"{chip_id}.tif"
        output = np.ones((512,512),dtype=np.uint8)
        output[prediction <= 0.5] = 0
        imsave(output_path, output)


if __name__ == '__main__':

    run_inference()
