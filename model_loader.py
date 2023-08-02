from transformers import AutoModelForImageClassification


class SwinModelLoader:
    def __init__(
        self, data_name="ImageNet", model_type="tiny", window_size=7, input_size=224
    ):
        if data_name.lower == "imagenet":
            self.define_imagenet_model_name(model_type, window_size, input_size)
        elif data_name.lower == "cifar10":
            self.define_cifar_model_name(data_name, model_type)
        else:
            Exception, "Model not supported"

    def define_imagenet_model_name(self, model_type, window_size, input_size):
        if model_type.lower() == "tiny":
            self.model_hugging_face_path = (
                f"microsoft/swin-tiny-patch4-window{window_size}-{input_size}"
            )
        elif model_type.lower() == "base":
            self.model_hugging_face_path = (
                f"microsoft/swin-base-patch4-window{window_size}-{input_size}"
            )
        elif model_type.lower() == "large":
            self.model_hugging_face_path = (
                f"microsoft/swin-large-patch4-window{window_size}-{input_size}"
            )

    def define_cifar_model_name(self, data_name):
        if data_name.lower == "cifar10":
            self.model_hugging_face_path = (
                "nielsr/swin-tiny-patch4-window7-224-finetuned-cifar10"
            )
        elif data_name.lower == "cifar100":
            self.model_hugging_face_path = (
                "jaycamper/swin-tiny-patch4-window7-224-finetuned-cifar100"
            )

    def load_load(self):
        model = AutoModelForImageClassification.from_pretrained(
            "nielsr/swin-tiny-patch4-window7-224-finetuned-cifar10"
        )
        model.eval()
        return model
