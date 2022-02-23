from pathlib import Path
import collections
from abc import ABC
import numpy as np
from tifffile import imsave, imread

from logger import Logger


class DataLoader(collections.Iterable, ABC):
    def __init__(self, root_dir, input_images_directory, mode="TEST"):
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        else:
            root_dir = root_dir
        self.mode = mode
        self.root_dir = root_dir
        self.idx = 0
        self.input_images_directory = input_images_directory if isinstance(input_images_directory, Path) else Path(
            input_images_directory)
        self.chip_ids = [pth.name for pth in input_images_directory.iterdir() if not pth.name.startswith(".")]

    def __next__(self):

        self.idx += 1
        if self.idx > len(self.chip_ids):
            self.idx = 0
            raise StopIteration
        try:

            chip_id = self.chip_ids[self.idx - 1]
            BANDS = ["B02", "B03", "B04", "B08"]
            band_arrs = []
            for band in BANDS:
                band_arr = imread(self.input_images_directory / f"{chip_id}/{band}.tif")
                band_arrs.append(band_arr)
            chip_arr = np.stack(band_arrs)
            # chip_arr = np.transpose(chip_arr, (1, 2, 0))
            chip_arr = chip_arr.reshape(1, 4, 512, 512)
            if self.mode == "TRAIN":
                true_data_path = self.root_dir / "train_labels"
                true_data_path = true_data_path / (chip_id + ".tif")

                true_data = imread(true_data_path)
                true_data = true_data.reshape(1, 1, 512, 512)
                return chip_arr, true_data
            return chip_arr
        except Exception as e:
            logger = Logger()
            logger.log("DataLoader" + str(e.args))
            return self.__next__()

    def __len__(self):
        return len(self.chip_ids)

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def random(self):
        import random
        idx_ = self.idx
        idx = random.randint(0, len(self.chip_ids))
        self.idx = idx
        res = self.__next__()
        self.idx = idx_
        return res


if __name__ == '__main__':
    pass
