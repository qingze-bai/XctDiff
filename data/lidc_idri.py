import torch
import numpy as np
import nibabel as nib
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from monai.data import load_decathlon_datalist

class LIDC_IDRI_Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self._loaditem(index)

    def _transform(self, im, is_drr=False):
        if is_drr:
            im = im.transpose(1, 0)
            im = im[:, ::-1]
            im = im / 255
        else:
            im = (im - im.min()) / (im.max() - im.min()) * 2 - 1.0
        im = np.expand_dims(im, axis=0)
        return torch.from_numpy(np.ascontiguousarray(im)).to(torch.float32)

    def _loaditem(self, index):
        dict = self.data[index]
        im = nib.load(dict['image'])
        filename = dict['xray'].split('/')[-1].split('.')[0]

        xray = Image.open(dict['xray'])
        xray = xray.convert("L")
        xray = xray.resize((128, 128))
        xray = np.array(xray)
        affine = im.affine
        return {
            "image" : self._transform(im.get_fdata()),
            "xray": self._transform(xray, is_drr=True),
            "affine": torch.from_numpy(affine).to(torch.float32),
            "filename": filename
        }

def get_loader(config):
    base_dir = config['base_dir']
    json_list = config['json_list']

    train_files = []
    for json in json_list:
        files = load_decathlon_datalist(json, False, "training", base_dir=base_dir)
        train_files += files

    val_files = []
    for json in json_list:
        files = load_decathlon_datalist(json, False, "validation", base_dir=base_dir)
        val_files += files

    print("Dataset all training: number of data: {}".format(len(train_files)))
    print("Dataset all validation: number of data: {}".format(len(val_files)))

    train_ds = LIDC_IDRI_Dataset(data=train_files)
    val_ds = LIDC_IDRI_Dataset(data=val_files)

    train_loader = DataLoader(train_ds,
                              batch_size=config["batch_size"],
                              shuffle=True,
                              num_workers=config.get("num_workers", 0),
                              pin_memory=config.get("pin_memory", False))

    val_loader = DataLoader(val_ds,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            num_workers=config.get("num_workers", 0),
                            pin_memory=config.get("pin_memory", False))

    return [train_loader, val_loader]