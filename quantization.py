from addict import Dict
from openvino.tools.pot.engines.ie_engine import IEEngine
from openvino.tools.pot.graph import load_model, save_model
from openvino.tools.pot.graph.model_utils import compress_model_weights
from openvino.tools.pot.pipeline.initializer import create_pipeline
from openvino.runtime import Core

from utils import CifarDataLoader, Accuracy


def configure_quantization():
    model_config = Dict(
        {
            "model_name": "mobilenet_v2",
            "model": "mobilenet_v2.xml",
            "weights": "mobilenet_v2.bin",
        }
    )
    engine_config = Dict(
        {"device": "CPU", "stat_requests_number": 2, "eval_requests_number": 2}
    )
    dataset_config = {"data_source": "data"}
    algorithms = [
        {
            "name": "DefaultQuantization",
            "params": {
                "target_device": "CPU",
                "preset": "performance",
                "stat_subset_size": 300,
            },
        }
    ]
    return model_config, engine_config, dataset_config, algorithms


def run_quantization(model_config, engine_config, dataset_config, algorithms, dataset):
    model = load_model(model_config)
    data_loader = CifarDataLoader(dataset_config, dataset)
    metric = Accuracy(top_k=1)
    engine = IEEngine(engine_config, data_loader, metric)
    pipeline = create_pipeline(algorithms, engine)
    compressed_model = pipeline.run(model)
    compress_model_weights(compressed_model)
    save_model(model=compressed_model, save_path=".", model_name="quantized_mobilenet")
    return pipeline, compressed_model


def evaluate_quantization(pipeline, model, compressed_model):
    metric_results = pipeline.evaluate(model)
    if metric_results:
        for name, value in metric_results.items():
            print(f"Accuracy of the original model: {name}: {value}")

    metric_results = pipeline.evaluate(compressed_model)
    if metric_results:
        for name, value in metric_results.items():
            print(f"Accuracy of the optimized model: {name}: {value}")

def benchmark_quantization():
    pass
    # # #
    # # !benchmark_app -m "mobilenet_v2.xml" -d CPU -api async
    # # !benchmark_app -m $compressed_model_xml -d CPU -api async


# ie = Core()

# # float model.
# float_model = ie.read_model(model="mobilenet_v2.xml", weights="mobilenet_v2.bin")
# float_compiled_model = ie.compile_model(model=float_model, device_name="CPU")

# # quantized model.
# quantized_model = ie.read_model(
#     model=compressed_model_xml, weights=compressed_model_bin
# )
# quantized_compiled_model = ie.compile_model(model=quantized_model, device_name="CPU")
