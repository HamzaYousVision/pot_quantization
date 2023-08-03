import torch
from torchvision import models
from transformers import AutoModelForImageClassification


class SwinModelLoader:
    def __init__(
        self, dataset_name="ImageNet", model_type="tiny", window_size=7, input_size=224
    ):
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.window_size = window_size
        self.input_size = input_size
        self.define_model_name()

    def define_model_name(self):
        if self.dataset_name.lower() == "imagenet":
            self.define_imagenet_model_name(
                self.model_type, self.window_size, self.input_size
            )
        elif "cifar" in self.dataset_name.lower():
            self.define_cifar_model_name(self.dataset_name)
        else:
            Exception, "Model not supported"

    def define_imagenet_model_name(self, model_type, window_size, input_size):
        author = "microsoft"
        model_type = model_type.lower()
        self.model_hugging_face_path = (
            f"{author}/swin-{model_type}-patch4-window{window_size}-{input_size}"
        )

    def define_cifar_model_name(self, dataset_name):
        if "10" in dataset_name.lower():
            self.model_hugging_face_path = (
                "nielsr/swin-tiny-patch4-window7-224-finetuned-cifar10"
            )
        elif "100" in dataset_name.lower():
            self.model_hugging_face_path = (
                "jaycamper/swin-tiny-patch4-window7-224-finetuned-cifar100"
            )

    def load_load(self):
        print(f"\n Loading swin transformer model trained on {self.dataset_name} ...")
        model = AutoModelForImageClassification.from_pretrained(
            self.model_hugging_face_path
        )
        print("\n model loaded")
        print(100 * "-", "\n")
        model.eval()
        return model


class MobilenetV2ModelLoader:
    def __init__(self, data_name="cifar10"):
        self.data_name = data_name

    def load_load(self):
        if "cifar" in self.data_name.lower():
            model_torch_paths = (
                "chenyaofo/pytorch-cifar-models",
                "cifar10_mobilenetv2_x1_0",
            )
            print("\n Loading mobilent v2 model trained on cifar10 ...")
            model = torch.hub.load(
                model_torch_paths[0], model_torch_paths[1], pretrained=True
            )
        elif self.data_name.lower() == "imagenet":
            print("\n Loading mobilent v2 model trained on imagenet ...")
            model = models.mobilenet_v2(pretrained=True)
        print("\n model loaded")
        print(100 * "-", "\n")
        model.eval()
        return model
