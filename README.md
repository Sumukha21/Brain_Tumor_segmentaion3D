# Brain_Tumor_segmentaion3D
## Project Objective:
  * Automate the localization and scanning of brain tumor regions in MRI scans using neural networks
  * Generate visualizations and insights from the segmentation prediction from the neural network
## Data Description:
The dataset contains multi-MRI scans acquired during a single imaging session. It includes the following types of scans:
  * Native T1-weighted (T1): This scan is obtained using a standard T1-weighted imaging sequence. This sequence highlights the differences in tissue types based on their contrast with the surrounding tissues
  * Post-contrast T1-weighted (T1Gd): This scan is obtained using a T1-weighted imaging sequence after the administration of a contrast agent such as Gadolinium. This sequence highlights the regions of the brain with a disrupted blood-brain barrier, such as enhancing tumor regions
  * T2-weighted (T2): This scan is obtained using a T2-weighted imaging sequence. This sequence highlights subtle differences in tissue types that are not visible on T1 scans
  * T2 Fluid Attenuated Inversion Recovery (T2-FLAIR): This scan is obtained using a T2-weighted imaging sequence. This sequence is useful for distinguishing between edema and other types of brain tissue
## Labels provided:
Given the MRI scans, we propose to localize and classify regions of the scans into the following categories:
  * label 0: No tumor
  * label 1: necrotic tumor core (Visible in T2): This class represents the core of the tumor, which is composed of necrotic tissue and non-enhancing tumor cells.
  * label 2: the peritumoral edematous/invaded tissue (Visible in flair): This class represents the edema, or swelling, that occurs around the tumor due to the accumulation of fluid in the surrounding brain tissue.
  * label 4: Gd-enhancing tumor (Needs to be converted to 3) (Visible in T1ce): This class represents the region of the tumor that enhances with the administration of contrast agent

## Code Desciption/Instructions:
  * If you want to just run the data preparation, analysis, model training, evaluation and visualization with UnetR or Vnet and the BRaTS2021 dataset, the brats_2021_segmentation_task.ipynb in the folder notebooks can be used.
  * The code also supports data preparation, analysis, model training, evaluation and visualization with custom models and datasets. The training can be performed as:
    - python source_code/main.py 
    - The path to the config file for training needs to be provided (Will be added as a command line argument in future update)
    - Sample configs for training (Unetr and Vnet available) can be used as reference for custom training pipelines (see source_code/configs/*)


