import os
import random 

import numpy as np
from sklearn.metrics import classification_report
import pandas as pd

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.RandomState(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True

seed_everything(42)

class BaseSolver:

    def __init__(self, path: str, path_valid=None, seed=42):
        
        self.path = path
        self.train = pd.read_json(path_or_buf=path, lines=True)
        self.seed = seed
        if path_valid:
            self.valid = pd.read_json(path_or_buf=path_valid, lines=True)
        else:
            self.valid = None
    
    def all_methods(self):

        if self.valid is not None:
            test_size = len(self.valid)
            y_true = list(self.valid.label)
        else:
            print("There are no Validation/Test set in this task")
            print("Making Predictions for Train dataset")
            test_size = len(self.train)
            y_true = self.train.label
            
        print()
        print(f"Making Prediction based on Majority Class")
        y_pred = self.majority_class(test_size=test_size)
        self.show_report(y_true, y_pred)

        print()
        print(f"Making Prediction based on Random Choice")
        y_pred = self.random_choice(test_size=test_size)
        self.show_report(y_true, y_pred)
        
        print()
        print(f"Making Prediction based on Random Choice Considered Classes Distribution")
        y_pred = self.random_balanced_choice(test_size=test_size)
        self.show_report(y_true, y_pred)


    def show_report(self, y_true, y_pred):
        print(classification_report(y_true, y_pred))
    
    
    def majority_class(self, test_size):
        """
        Make prediction based on majority class of train dataset
        test_size: how many predictions should be made
        return: List of predictions
        """

        prediction = self.train.label.mode()[0]
        y_pred = [prediction] * test_size
        return y_pred

    def random_choice(self, test_size):
        """
        Make random predictions
        label: label column in df (str)
        test_size: how many predictions should be made
        return: List of predictions
        """
        options = sorted(self.train.label.unique())
        np.random.seed(self.seed)
        y_pred = np.random.choice(options, size=test_size)
        return y_pred


    def random_balanced_choice(self, test_size):
        """
        Make random predictions with calculated probabilities
        label: label column in df (str)
        test_size: how many predictions should be made
        return: List of predictions
        """
        frequences = dict(self.train.label.value_counts(normalize=True))

        labels = []
        probs = []
        for key, value in frequences.items():
            labels.append(key)
            probs.append(value)
        np.random.seed(self.seed)
        y_pred = np.random.choice(labels, size=test_size, p=probs)
        return y_pred