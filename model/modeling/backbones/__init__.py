__all__ = ["build_backbone"]


def build_backbone(config, model_type):
    # 判断训练模式，目前只关注det，backbone只有resnet
    if model_type == "det":
        from .ResNet import ResNet
        support_dict = ["ResNet"]
    module_name = config.pop('name')
    assert module_name in support_dict, \
        Exception("when model typs is {}, backbone only support {}"
                  .format(model_type, support_dict))
    model_class = eval(module_name)(**config)
    return model_class
