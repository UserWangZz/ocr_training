__all__ = ['build_head']


def build_head(config):
    from .det_db_head import DBHead
    support_dict = ['DBHead']

    module_name = config.pop('name')
    assert module_name in support_dict, \
        Exception('head only support {}'.format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class
