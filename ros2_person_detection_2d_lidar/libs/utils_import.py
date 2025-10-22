"""
Utility module to handle dynamic imports.
This module allows to search for datasets, models, etc., in this repo and in external modules.
Classes from external modules may be overlayered by adding a class with the same name in this repo without requiring to change names in configs or in imports throughout the codebase.
"""

import inspect


def import_model(name):
    """Return model class or factory function"""
    import ros2_person_detection_2d_lidar.models as custom_models

    modules = [custom_models]

    for module in modules:
        if hasattr(module, name):
            class_or_function_found = getattr(module, name)
            if inspect.isclass(class_or_function_found) or inspect.isfunction(class_or_function_found):
                return class_or_function_found

    raise ImportError(f"Model class or factory function {name} not found")


def import_module(name):
    """Return module class or factory function."""
    import ros2_person_detection_2d_lidar.models as custom_models
    import torch.nn as torch_nn

    modules = [custom_models, torch_nn]

    for module in modules:
        if hasattr(module, name):
            class_or_function_found = getattr(module, name)
            if inspect.isclass(class_or_function_found) or inspect.isfunction(class_or_function_found):
                return class_or_function_found

    raise ImportError(f"Module {name} not found")
