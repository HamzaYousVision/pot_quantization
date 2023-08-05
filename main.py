import os
import argparse

from model_loader import SwinModelLoader, MobilenetV2ModelLoader
from ov_model_optimizer import OpenVinoModelOptimizer
from dataset_loader import DatasetLoader
from pot_quantizer import PotQuantizer
from evaluator import Evaluator


def create_folders():
    if not os.path.exists("model"):
        os.makedirs("model")

    if not os.path.exists("data"):
        os.makedirs("data")


def main(args):
    # perepare folders
    create_folders()

    # define model
    if "swin" in args.model_name.lower():
        model_loader = SwinModelLoader(args.dataset_name)
    elif "mobilenet" in args.model_name.lower():
        model_loader = MobilenetV2ModelLoader(args.dataset_name)
    model = model_loader.load_load()

    # generate openvino IR
    ov_model_optimizer = OpenVinoModelOptimizer(
        model, args.model_name, args.dataset_name
    )
    ov_model_optimizer.export_to_onnx()
    ov_model_optimizer.run_model_optimizer()

    # load dataset
    dataset_loader = DatasetLoader(args.dataset_name)
    dataset = dataset_loader.load_dataset()

    # create and run openvino POT compression pipeline
    model_ir_files = ov_model_optimizer.get_model_ir_files()
    quantizer = PotQuantizer(model_ir_files, dataset)
    quantizer.configure(args)
    quantizer.create_pipeline()
    quantizer.run_pipeline()
    quantizer.compress_optimized_model()
    quantizer.save_optimized_model(args.model_name)

    # evaluate original/compressed models
    pipeline = quantizer.pipeline
    fp32_model = quantizer.model
    quantized_model = quantizer.model_quantized
    evaluator = Evaluator(pipeline)

    evaluator.evaluate_accuracy(fp32_model, "original")
    evaluator.evaluate_accuracy(quantized_model, "quantized")

    model_ir_xml = model_ir_files[0]
    quantized_model_ir_xml, _ = quantizer.get_quantized_model_ir_files()
    evaluator.evaluate_FPS(model_ir_xml)
    evaluator.evaluate_FPS(quantized_model_ir_xml)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="POT Quantization Script")
    parser.add_argument("--model_name", type=str, default="swin", help="model name")
    parser.add_argument(
        "--dataset_name", type=str, default="ImageNet", help="model name"
    )
    parser.add_argument(
        "--exclude_MVN",
        action="store_true",
        help="Option to exclude LayerNorm from quantization",
    )

    args = parser.parse_args()

    main(args)
