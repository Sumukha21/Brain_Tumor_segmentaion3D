import nibabel as nib
import glob
import os
import torchvision.transforms as transforms
import torch
import random
from source_code.models.unetr import UNETR
from source_code.utilities.utils import text_file_writer


if __name__ == "__main__":
    dataset_source_folder = "dataset/brats_train_10_data"
    sample_size = 10
    save_path = "dataset/lists/sample_%d.txt" % sample_size
    patient_folders_list = sorted(os.listdir(dataset_source_folder))
    patient_folders_list = [os.path.join(dataset_source_folder, i) for i in patient_folders_list]
    sample_patient_folder_list = random.sample(patient_folders_list, sample_size)
    text_file_writer(save_path, sample_patient_folder_list)


if __name__ == "__main1__":
    model = UNETR(
            in_channels=3,
            out_channels=4,
            img_size=(144, 240, 240),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            conv_block=True,
            res_block=True,
            dropout_rate=0,
        )
    transformation = transforms.ToTensor()
    img_folder = "C:/Users/sumuk/OneDrive/Desktop/NCSU_related/Courses_and_stuff/Courses_and_stuff/NCSU_courses_and_books/ECE_542/FinalProj/Brain_Tumor_segmentaion3D/dataset/BRaTS2021_Training_Data/BraTS2021_00000"
    flair_path = glob.glob(os.path.join(img_folder, "*_flair.nii.gz"))[0]
    t1ce_path = glob.glob(os.path.join(img_folder, "*t1ce.nii.gz"))[0]
    t2_path = glob.glob(os.path.join(img_folder, "*_t2.nii.gz"))[0]
    flair = transformation(nib.load(flair_path).get_fdata()[:, :, 5: -6])
    t1ce = transformation(nib.load(t1ce_path).get_fdata()[:, :, 5: -6])
    t2 = transformation(nib.load(t2_path).get_fdata()[:, :, 5: -6])
    stacked = torch.stack([flair, t1ce, t2])
    stacked = stacked.unsqueeze(0)
    stacked = stacked.to(torch.float32)
    prediction = model(stacked)
    print("")




