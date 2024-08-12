import importlib

def get_dataset(dataset_class):
    dataset_class_name = dataset_class
    module_name = f'src.data.{dataset_class_name.lower()}'
    module = importlib.import_module(module_name)
    dataset_class = getattr(module, dataset_class_name)
    return dataset_class