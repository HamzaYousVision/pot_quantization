import os
import torch
import subprocess


class OpenVinoModelOptimizer:
    def __init__(self, model, model_name, dataset_name):
        self.model = model
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.define_onnx_model_path()
        self.define_dummy_input()

    def define_onnx_model_path(self):
        model_root = "model"
        if "swin" in self.model_name.lower():
            self.onnx_model_path = os.path.join(model_root, "swin.onnx")
        else:
            self.onnx_model_path = os.path.join(model_root, "mobilenet_v2.onnx")

    def define_dummy_input(self):
        if "swin" in self.model_name.lower():
            self.dummy_input = torch.randn(1, 3, 224, 224)
            self.input_shape = "[1,3,224,224]"
        elif "mobilenet" in self.model_name.lower():
            if "imagenet" in self.dataset_name.lower():
                self.dummy_input = torch.randn(1, 3, 224, 224)
                self.input_shape = "[1,3,224,224]"
            elif "cifar" in self.dataset_name.lower():
                self.dummy_input = torch.randn(1, 3, 32, 32)
                self.input_shape = "[1,3,32,32]"

    def export_to_onnx(self):
        torch.onnx.export(
            self.model, self.dummy_input, self.onnx_model_path, verbose=True
        )

    def run_model_optimizer(self):
        command = f"mo --framework=onnx --input_shape={self.input_shape} -m {self.onnx_model_path} --output_dir model"
        print(f"\n running model optimizer on {self.onnx_model_path} ...")
        subprocess.check_output(command, shell=True)
        print("\n Openvino IR (xml and bin files) is created")
        print(100 * "-", "\n")

    def get_model_ir_files(self):
        ir_model_xml = self.onnx_model_path.replace(".onnx", ".xml")
        ir_model_bin = self.onnx_model_path.replace(".onnx", ".bin")
        return (ir_model_xml, ir_model_bin)
