import torch
import nibabel as nib
import numpy as np
import os
import glob
import torchvision.transforms as transforms
from source_code.utilities.utils import text_file_reader


class BTSDataset(torch.utils.data.Dataset):
    def __init__(self, patient_data_list_path, no_classes=4):
        patient_data_list = text_file_reader(patient_data_list_path)
        self.patient_flair_scans_list = [glob.glob(os.path.join(i, "*_flair.nii.gz"))[0] for i in patient_data_list]
        self.patient_t1ce_scans_list = [glob.glob(os.path.join(i, "*_t1ce.nii.gz"))[0] for i in patient_data_list]
        self.patient_t2_scans_list = [glob.glob(os.path.join(i, "*_t2.nii.gz"))[0] for i in patient_data_list]
        self.patient_seg_scans_list = [glob.glob(os.path.join(i, "*_seg.nii.gz"))[0] for i in patient_data_list]
        self.transform = transforms.ToTensor()
        self.no_classes = no_classes

    def __len__(self):
        return len(self.patient_flair_scans_list)

    def __getitem__(self, idx):
        t1ce_scan = self.transform(np.asarray(nib.load(self.patient_t1ce_scans_list[idx]).get_fdata())[:, :, 5: -6])
        t2_scan = self.transform(np.asarray(nib.load(self.patient_t2_scans_list[idx]).get_fdata())[:, :, 5: -6])
        flair_scan = self.transform(np.asarray(nib.load(self.patient_flair_scans_list[idx]).get_fdata())[:, :, 5: -6])
        seg_label = np.asarray(nib.load(self.patient_seg_scans_list[idx]).get_fdata()[:, :, 5: -6])
        seg_label[seg_label == 4] = 3
        seg_label = self.transform(seg_label)
        seg_label_ohe = torch.nn.functional.one_hot(seg_label.to(torch.int64), self.no_classes)
        seg_label_ohe = torch.moveaxis(seg_label_ohe, -1, 0)
        image_scans_stacked = torch.stack([t1ce_scan, t2_scan, flair_scan])
        return image_scans_stacked.to(torch.float32), seg_label_ohe.to(torch.float32)


if __name__ == "__main1__":
    path = ""
    folder_list = sorted(glob.glob(os.path.join(path, "*")))
    folder_list = [i for i in folder_list if os.path.isdir(i)][: 20]
    sample_dataset = BTSDataset(folder_list)


