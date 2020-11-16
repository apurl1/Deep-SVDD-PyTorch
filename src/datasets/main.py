from .mnist import MNIST_Dataset
from .fashion import Fashion_Dataset
from .cifar10 import CIFAR10_Dataset


def load_dataset(dataset_name, data_path, outlier_class):
    """Loads the dataset."""

    implemented_datasets = ('mnist', 'fashion')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path, outlier_class=outlier_class)
    
    if dataset_name == 'fashion':
        dataset = Fashion_Dataset(root=data_path, outlier_class=outlier_class)

    #if dataset_name == 'cifar10':
        #dataset = CIFAR10_Dataset(root=data_path, outlier_class=outlier_class)

    return dataset
