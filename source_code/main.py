import torch.cuda
import yaml
from data_generator.data_generator import BTSDataset
from models.unetr import UNETR
from Training_Evaluation.trainer import Trainer
from Training_Evaluation.evaluator import generate_predictions
from source_code.utilities.utils import instantiate_class

# device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main1__":
    config_path = "source_code/configs/overfitting_test.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    model = UNETR(**config["Model"])
    training_datagenerator = BTSDataset(**config["Training_Dataset"])
    validation_datagenerator = BTSDataset(**config["Validation_Dataset"])
    test_datagenerator = BTSDataset(**config["Test_Dataset"])
    training_dataloader = torch.utils.data.DataLoader(training_datagenerator, batch_size=1,
                                                      shuffle=True)
    trainer = Trainer(**config["Trainer"], model=model, weights_save_folder=config["save_folder"])
    model_weights, trained_model = trainer(training_dataloader, training_dataloader)
    print("")


if __name__ == "__main2__":
    # config_path = "/content/drive/MyDrive/Personal/MS/Brain_Tumor_segmentaion3D/source_code/configs/overfitting_test.yaml"
    config_path = "/content/drive/MyDrive/ECE542/Brain_Tumor_segmentaion3D/source_code/configs/unetr_fullscale_training_ncsu.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    model = UNETR(**config["Model"])
    training_datagenerator = BTSDataset(**config["Training_Dataset"])
    training_dataloader = torch.utils.data.DataLoader(training_datagenerator, batch_size=3,
                                                      shuffle=True)
    validation_datagenerator = BTSDataset(**config["Validation_Dataset"])
    validation_dataloader = torch.utils.data.DataLoader(validation_datagenerator, batch_size=3,
                                                        shuffle=False)
    trainer = Trainer(**config["Trainer"], model=model, weights_save_folder=config["save_folder"])
    model_weights, trained_model = trainer(training_dataloader, validation_dataloader)
    print("")
