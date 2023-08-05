from addict import Dict
from openvino.tools.pot.engines.ie_engine import IEEngine
from openvino.tools.pot.graph import load_model, save_model
from openvino.tools.pot.graph.model_utils import compress_model_weights
from openvino.tools.pot.pipeline.initializer import create_pipeline
from openvino.runtime import Core

from ov_utils import DataLoader, Accuracy


class PotQuantizer:
    def __init__(self, model_ir_files, dataset):
        self.ir_model_xml = model_ir_files[0]
        self.ir_model_bin = model_ir_files[1]
        self.dataset = dataset

    def configure(self, args, device="CPU"):
        self.model_config = Dict(
            {
                "model_name": args.model_name,
                "model": self.ir_model_xml,
                "weights": self.ir_model_bin,
            }
        )
        self.engine_config = Dict(
            {"device": device, "stat_requests_number": 2, "eval_requests_number": 2}
        )
        self.dataset_config = {"data_source": "data"}
        self.configure_quantization_pipeline(args.exclude_MVN, device)

    def configure_quantization_pipeline(self, exclude_MVN, device):
        params = {
            "target_device": device,
            "preset": "performance",
            "stat_subset_size": 300,
        }
        if exclude_MVN:
            params.update({"model_type": "transformer"})

        self.algorithms = [{"name": "DefaultQuantization", "params": params}]

    def create_pipeline(self):
        self.model = load_model(self.model_config)
        data_loader = DataLoader(self.dataset_config, self.dataset)
        metric = Accuracy(top_k=1)
        engine = IEEngine(self.engine_config, data_loader, metric)
        self.pipeline = create_pipeline(self.algorithms, engine)

    def run_pipeline(self):
        print(f"\n running quantization ...")
        self.model_quantized = self.pipeline.run(self.model)
        print("\n Model quantized")
        print(100 * "-", "\n")

    def compress_optimized_model(self):
        print(f"\n compressing model weights ...")
        compress_model_weights(self.model_quantized)
        print("\n Weight are compressed")
        print(100 * "-", "\n")

    def save_optimized_model(self, model_name):
        print(f"\n saving model ...")
        self.optimized_model_name = f"quantized_{model_name}"
        save_model(
            model=self.model_quantizer,
            save_path="model",
            model_name=self.optimized_model_name,
        )
        print("\n Model saved")
        print(100 * "-", "\n")

    def get_quantized_model_ir_files(self):
        ir_model_xml = f"{self.optimized_model_name}.xml"
        ir_model_bin = f"{self.optimized_model_name}.bin"
        return ir_model_xml, ir_model_bin
