import os, sys
import warnings; warnings.filterwarnings("ignore")
# import pytorch_lightning; pytorch_lightning.seed_everything(22)

import collections
import glob
import tqdm

import argparse
import pandas, numpy as np
import torch
import torch.nn as nn, torch.optim as optim
import torch.nn.functional as F
import flwr as fl
import wandb

class FedAvg(fl.server.strategy.FedAvg):
    def __init__(self, 
        initial_model, 
        trainer,
        test_dataloader,
        *args, **kwargs
    ):
        self.trainer = trainer
        self.initial_model = initial_model
        self.test_dataloader = test_dataloader
        super().__init__(*args, **kwargs)

    def aggregate_fit(self, 
        server_round, 
        results, failures, 
    ):

        aggregated_parameters, results = super().aggregate_fit(
            server_round, 
            results, failures, 
        )
        if aggregated_parameters is not None:
            self.initial_model.load_state_dict(
                collections.OrderedDict(
                    {key:torch.tensor(value) for key, value in zip(self.initial_model.state_dict().keys(), fl.common.parameters_to_ndarrays(aggregated_parameters))}
                ), 
                strict = True, 
            )
            self.trainer.fit(self.initial_model, self.test_dataloader)

        return aggregated_parameters, {}