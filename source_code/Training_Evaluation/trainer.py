from tqdm import tqdm
import torch
import os
import numpy as np
import gc
from source_code.utilities.utils import check_path, instantiate_class, instantiate_attribute


device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["KMP_DUPLICATE_LIB_OK"] = 'True'


class Trainer:
    def __init__(self, model, loss, optimizer, number_of_epochs, weights_save_folder):
        self.loss = instantiate_class(**loss)
        self.number_of_epochs = number_of_epochs
        self.model = model
        self.model.to("cuda")
        self.optimizer = instantiate_attribute(optimizer["path"])(self.model.parameters(), **optimizer["params"])
        self.best_model_weights = None
        self.weights_save_folder = weights_save_folder
        check_path(self.weights_save_folder)

    def __call__(self, training_dataloader, validation_dataloader):
        average_val_loss = 10 ** 6
        for i in range(self.number_of_epochs):
            print("Epoch number: ", i)
            training_loss = []
            for image, label in tqdm(training_dataloader):
                self.optimizer.zero_grad()
                image = image.cuda().to(device)
                probabilities = self.model(image)
                loss = self.loss(probabilities, label.cuda().to(device))
                loss.backward()
                training_loss.append(loss.detach().cpu())
                self.optimizer.step()
            print("Average training loss: ", np.average(training_loss))
            validation_loss = []
            for image, label in tqdm(validation_dataloader):
                with torch.no_grad():
                    image = image.cuda().to(device)
                    probabilities = self.model(image)
                    validation_loss.append(self.loss(probabilities, label.cuda().to(device)).cpu())
            print("Average validation loss: ", np.average(validation_loss))
            if np.average(validation_loss) < average_val_loss:
                self.best_model_weights = self.model.state_dict()
                average_val_loss = np.average(validation_loss)
                torch.save(self.model.state_dict(), os.path.join(self.weights_save_folder, 'model_weights_%d.pth' % i))
        return self.best_model_weights, self.model.to("cpu")

