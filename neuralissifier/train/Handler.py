import torch.nn as nn
import torch.utils.data as td
import torch.optim as to
import pandas as pd


class Handler:
    def __init__(
            self,
            model: nn.Module,
            criterion: nn.modules.loss._Loss,
            dataloader: td.DataLoader,
            optimizer: to.Optimizer,
            epochs: int,
            mode: str = "train",
            metrics: list = None,
            device: str = "cpu",
            params: dict = {}
    ) -> None:
        """
            Model: Pytorch Model object
            Criterion: Pytorch Criterion object
            Dataloader: Pytorch Dataloader object
            Optimizer: Pytorch Optimizer object
            Epochs (int): Number of Epochs to Run
            Mode (str): "train" (with full backprop and metrics),
                        "test" (no backprop, compute metrics),
                        "infer" (no backprop, no metrics)
            Metrics (list): List of metrics objects which have a
                                .compute function and a name field
            Device (str): Pytorch device to use
        """
        self.model = model
        self.criterion = criterion
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.epochs = epochs
        self.metrics = metrics
        self.current_epoch = 0
        self.current_batch = 0
        self.mode = mode.lower()
        self.results = None
        self.device = device
        self.params = params

    def run(self) -> pd.DataFrame:
        """
            Run the full training experience.
                Return the list as a DataFrame.
        """
        self.current_epoch = 0
        results = []
        while self.current_epoch < self.epochs:
            self.handle_epoch()
        result_df = pd.DataFrame(results)
        self.results = result_df
        return result_df

    def handle_epoch(self) -> list:
        """
            Run a single epoch
        """
        epoch_result = []
        for index, batch in enumerate(self.dataloader):
            self.current_batch = index
            batch_result = self.handle_batch(batch)
            epoch_result.append(batch_result)
        self.current_epoch += 1
        return epoch_result

    def handle_batch(self, batch) -> dict:
        """
            Handle a single batch
                return the loss float
        """
        image, label = batch
        inference = self.model(image)
        loss = 0.
        batch_result = dict(
            **self.params,
            index=self.current_batch,
            epoch=self.current_epoch
        )
        if self.criterion is not None:
            loss = self.criterion(inference, label)
            if "train" in self.mode:
                loss.backward()
            batch_result['loss'] = loss.item()
        for metric in self.metrics:
            value = metric.compute(image, label, inference, self.model)
            batch_result[metric.name] = value
        return batch_result
