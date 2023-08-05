import subprocess


class Evaluator:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def evaluate_accuracy(self, model, model_name):
        print(f"\n running accuracy evaluation of the {model_name} model ...")
        metric_results = self.pipeline.evaluate(model)
        if metric_results:
            for name, value in metric_results.items():
                print(f"Accuracy of the {model_name} model: {name}: {value}")
        print(100 * "-", "\n")

    def evaluate_FPS(self, model_xml):
        print(f"\n running accuracy evaluation of the {model_xml} model ...")
        command = f"benchmark_app -m {model_xml} -d CPU -api async"
        subprocess.check_output(command, shell=True)
        print(100 * "-", "\n")
