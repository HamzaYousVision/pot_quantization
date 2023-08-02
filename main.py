from model_loader import SwinModelLoader
from openv_vino_model_optimizer import OpenVinoModelOptimizer
from dataset_loader import DatasetLoader
from pot_quantizer import PotQuantizer


def main():
    #
    model_loader = SwinModelLoader()
    model = model_loader.load_load()

    #
    ov_model_optimizer = OpenVinoModelOptimizer(model)
    ov_model_optimizer.export_to_onnx()
    ov_model_optimizer.run_model_optimizer()

    #
    dataset_loader = DatasetLoader()
    dataset = dataset_loader.load_imagenet_dataset()

    #
    model_ir_files = ov_model_optimizer.get_model_ir_files()
    quantizer = PotQuantizer(model_ir_files, dataset)
    quantizer.configure()
    quantizer.create_pipeline()
    quantizer.run_pipeline()

    #
    quantizer.evaluate_fp32_model()
    quantizer.evaluate_quantized_model()


if __name__ == "__main__":
    main()
