import numpy as np
from addict import Dict
from openvino.tools.pot.api import DataLoader, Metric


class DataLoader(DataLoader):
    def __init__(self, config, dataset):
        if not isinstance(config, Dict):
            config = Dict(config)
        super().__init__(config)
        self.indexes, self.pictures, self.labels = self.load_data(dataset)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        return (self.indexes[index], self.labels[index]), self.pictures[index].numpy()

    def load_data(self, dataset):
        pictures, labels, indexes = [], [], []
        for idx, sample in enumerate(dataset):
            pictures.append(sample[0])
            labels.append(sample[1])
            indexes.append(idx)
        return indexes, pictures, labels


class Accuracy(Metric):
    def __init__(self, top_k=1):
        super().__init__()
        self._top_k = top_k
        self._name = "accuracy@top{}".format(self._top_k)
        self._matches = []

    @property
    def value(self):
        return {self._name: self._matches[-1]}

    @property
    def avg_value(self):
        return {self._name: np.ravel(self._matches).mean()}

    def update(self, output, target):
        if len(output) > 1:
            raise Exception(
                "The accuracy metric cannot be calculated "
                "for a model with multiple outputs"
            )
        if isinstance(target, dict):
            target = list(target.values())
        predictions = np.argsort(output[0], axis=1)[:, -self._top_k :]
        match = [float(t in predictions[i]) for i, t in enumerate(target)]

        self._matches.append(match)

    def reset(self):
        self._matches = []

    def get_attributes(self):
        return {self._name: {"direction": "higher-better", "type": "accuracy"}}
