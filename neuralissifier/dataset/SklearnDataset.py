import torch
from torch.utils.data import Dataset
import sklearn.datasets as skd

class SklearnDataset(Dataset):
    def __init__(self, function, *args, device="cpu", **kwargs):
        super(SklearnDataset, self).__init__()
        batch = function(*args, **kwargs)        
        if type(batch) is tuple and len(batch) == 2:
            self.X, self.y = batch        
        elif type(batch) is tuple and len(batch) == 1:
            self.X = self.y = batch[0]
        elif type(batch) is tuple and len(batch) == 3:
            self.X = batch[0]
            self.y = batch[1].T
        else:
            if hasattr(batch, 'data'):
                self.X = batch.data
            else:
                raise(NotImplemented("Type is %s" % str(type(batch))))
            if hasattr(batch, 'target'):
                self.y = batch.target
            else:
                self.y = self.X
        self.X = torch.Tensor(self.X).to(device)

        self.y = torch.Tensor(self.y).to(device)

    def __getitem__(self, k):
        return self.X[k, ...], self.y[k,...]

    def __len__(self):
        return self.y.shape[0]
    
class SKLTwentyNewsgroups(SklearnDataset):
    def __init__(self, *args, device="cpu", **kwargs):
        super(SKLTwentyNewsgroups, self).__init__(skd.fetch_20newsgroups, *args, device=device, **kwargs)

class SKLBreastCancer(SklearnDataset):
    def __init__(self, *args, device="cpu", **kwargs):
        super(SKLBreastCancer, self).__init__(skd.load_breast_cancer, *args, return_X_y=True, **kwargs)

class SKLDiabetes(SklearnDataset):
    def __init__(self, *args, device="cpu", **kwargs):
        super(SKLDiabetes, self).__init__(skd.load_diabetes, *args, return_X_y=True, **kwargs)

class SKLDigits(SklearnDataset):
    def __init__(self, *args, device="cpu", **kwargs):
        super(SKLDigits, self).__init__(skd.load_digits, *args, return_X_y=True, **kwargs)

class SKLIris(SklearnDataset):
    def __init__(self, *args, device="cpu", **kwargs):
        super(SKLIris, self).__init__(skd.load_iris, *args, return_X_y=True, **kwargs)

class SKLLinnerud(SklearnDataset):
    def __init__(self, *args, device="cpu", **kwargs):
        super(SKLLinnerud, self).__init__(skd.load_linnerud, *args, return_X_y=True, **kwargs)

class SKLWine(SklearnDataset):
    def __init__(self, *args, device="cpu", **kwargs):
        super(SKLWine, self).__init__(skd.load_wine, *args, return_X_y=True, **kwargs)

class SKLBicluster(SklearnDataset):
    def __init__(self, *args, device="cpu", **kwargs):
        super(SKLBicluster, self).__init__(skd.make_biclusters, *args, **kwargs)

class SKLBlobs(SklearnDataset):
    def __init__(self, *args, device="cpu", **kwargs):
        super(SKLBlobs, self).__init__(skd.make_blobs, *args, **kwargs)

if __name__=="__main__":
    dataclasses = [SKLBreastCancer, SKLDiabetes,SKLDigits,SKLIris,SKLLinnerud,SKLWine,SKLBicluster,SKLBlobs]
    datakwargs = [{},{},{},{},{},{},{}, {"centers":4}]
    dataargs = [[],[],[],[],[],[],[(100,100),4], [100,100]]
    for dataclass, args, kwargs in zip(dataclasses, dataargs, datakwargs):
        print(dataclass.__name__)
        dataset = dataclass(*args, **kwargs)
        print("\t", len(dataset))
        print("\t", dataset.X.shape, dataset.y.shape)