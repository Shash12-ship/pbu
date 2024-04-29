import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler



if torch.cuda.is_available() == True:
    device = 'cuda:1'    
else:
    device = 'cpu'


def testdatasplit(class_number, testdata):
    """Test data splitting"""
    # Test dataloader with 3's only
    test_threes_index = []
    test_nonthrees_index = []
    for i in range(0, len(testdata)):
      if testdata.targets[i] == class_number:
        test_threes_index.append(i)
      else:
        test_nonthrees_index.append(i)
    three_test_loader = DataLoader(testdata, batch_size=64,
                  sampler = SubsetRandomSampler(test_threes_index))
    nonthree_test_loader = DataLoader(testdata, batch_size=64,
                  sampler = SubsetRandomSampler(test_nonthrees_index))
    
    return three_test_loader, nonthree_test_loader


def compute_unlearned_predictions(model, three_test_loader, nonthree_test_loader):
    model.eval()
    with torch.no_grad():
        # For three_test_loader
        correct_three = 0
        total_three = 0
        for inputs, labels in three_test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_three += labels.size(0)
            correct_three += (predicted == labels).sum().item()

        accuracy_three = correct_three / total_three
        print(f"Accuracy on Three: {accuracy_three * 100:.2f}%")

        # For nonthree_test_loader
        correct_nonthree = 0
        total_nonthree = 0
        for inputs, labels in nonthree_test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_nonthree += labels.size(0)
            correct_nonthree += (predicted == labels).sum().item()

        accuracy_nonthree = correct_nonthree / total_nonthree
        print(f"Accuracy on Non-Three: {accuracy_nonthree * 100:.2f}%")

        return accuracy_three, accuracy_nonthree