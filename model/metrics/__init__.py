import copy

__all__ = ["build_metric"]

from .det_metric import DetMetric, DetFCEMetric


def build_metric(config):
    support_dict = [
        "DetMetric", "DetFCEMetric"
    ]

    config = copy.deepcopy(config)
    module_name = config.pop("name")
    assert module_name in support_dict, Exception(
        "metric only support {}".format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class
