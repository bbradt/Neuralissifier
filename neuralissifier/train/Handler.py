import torch.nn as nn
import torch.utils.data as td
import torch.optim as to


class Handler:
    def __init__(
            self,
            model: nn.Module,
            criterion: nn.modules.loss._Loss,
            dataloader: td.DataLoader,
            optimizer: to.Optimizer,
            epochs: int,
            mode: str = "train",
            metrics: list = None
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.epochs = epochs
        self.metrics = metrics
        self.current_epoch = 0
        self.current_batch = 0
        self.mode = mode.lower()
        self.results = []

    def run(self):
        self.current_epoch = 0
        while self.current_epoch < self.epochs:
            for index, batch in enumerate(self.dataloader):
                self.current_batch = index
                self.handle_batch(batch)

    def handle_batch(self):
        pass
