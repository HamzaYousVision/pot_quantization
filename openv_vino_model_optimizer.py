import os
import torch
import subprocess


class OpenVinoModelOptimizer:
    def __init__(self, model):
        self.model = model

    def export_to_onnx(self):
        dummy_input = torch.randn(1, 3, 224, 224)
        self.onn_model_path = os.path.join("model", "swin.onnx")
        torch.onnx.export(self.model, dummy_input, self.onn_model_path, verbose=True)

    def run_model_optimizer(self):
        command = f"mo --framework=onnx --input_shape=[1,3,224,224] -m {self.onn_model_path} --output_dir model"
        print(f"\n running model optimizer on {self.onn_model_path} ...")
        subprocess.check_output(command, shell=True)

    def get_model_ir_files(self):
        ir_model_xml = self.onnx_model_path.with_suffix(".xml")
        ir_model_bin = self.onnx_model_path.with_suffix(".bin")
        return (ir_model_xml, ir_model_bin)
