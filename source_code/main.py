import torch.cuda
import yaml
from data_generator.data_generator import BTSDataset
from models.unetr import UNETR
from Training_Evaluation.trainer import Trainer
from Training_Evaluation.evaluator import generate_predictions
from source_code.utilities.utils import instantiate_class

# device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    config_path = "source_code/configs/overfitting_test.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    model = UNETR(**config["Model"])
    training_datagenerator = BTSDataset(**config["Training_Dataset"])
    validation_datagenerator = BTSDataset(**config["Validation_Dataset"])
    test_datagenerator = BTSDataset(**config["Test_Dataset"])
    training_dataloader = torch.utils.data.DataLoader(training_datagenerator, batch_size=4,
                                                      shuffle=True)
    trainer = Trainer(**config["Trainer"], model=model, weights_save_folder=config["save_folder"])
    model_weights, trained_model = trainer(training_dataloader, training_dataloader)
    print("")
