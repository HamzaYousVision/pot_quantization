import os
from torchvision import transforms, datasets


class DatasetLoader:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def load_cifar10_dataset(self):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        )
        dataset = datasets.CIFAR10(
            root="data", train=False, transform=transform, download=True
        )
        return dataset

    def load_imagenet_dataset(self):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        transform = transforms.Compose(
            [
                transforms.Resize(246),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )
        data_path = os.path.join("data", "val")
        dataset = datasets.ImageFolder(data_path, transform)
        return dataset

    def load_dataset(self):
        print(f"\n Loading {self.dataset_name} dataset ...")
        if "cifar" in self.dataset_name.lower():
            dataset = self.load_cifar10_dataset()
        elif "imagenet" in self.dataset_name.lower():
            dataset = self.load_imagenet_dataset()
        print("\n dataset loaded")
        print(100 * "-", "\n")
        return dataset
