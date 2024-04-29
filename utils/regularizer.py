import numpy as np
import torch
import torch.nn as nn
from nngeometry.metrics import FIM
from nngeometry.object import PMatDiag, PVector


if torch.cuda.is_available() == True:
    device = 'cuda:1'    
else:
    device = 'cpu'



class PBU_reg(object):

    def __init__(self, init_model: nn.Module, unlearned_model: nn.Module, train_loader,
                 Representation, device='cuda:1'):
        self.device = device
        self.init_model = init_model
        self.unlearned_model = unlearned_model
        self.train_loader = train_loader
        self.Fisher, self.v0 = self.compute_fisher(self.init_model, self.train_loader, Representation)

    def compute_fisher(self, model, loader, Representation):
        # Set BatchNorm layers to evaluation mode
        self.init_model.train()
        for module in self.init_model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

        F_diag = FIM(model=self.init_model,
                     loader=self.train_loader,
                     representation=Representation,
                     n_output=10,
                     variant='classif_logits',
                     device=self.device)

        # Re-enable BatchNorm for training
        self.init_model.train()
        for module in self.init_model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train()

        v0 = PVector.from_model(self.init_model).clone()
        return F_diag, v0

    def penalty(self):
        v = PVector.from_model(self.unlearned_model)
        regularization_loss = self.Fisher.vTMv(v - self.v0)
        return regularization_loss