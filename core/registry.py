from typing import Any, Dict, Optional, Type, Callable, Union

class Registry:
    """A registry to map strings to classes.

    Args:
        name (str): Registry name.
    """

    def __init__(self, name: str):
        self._name = name
        self._module_dict: Dict[str, Type] = {}

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self._name}, items={self._module_dict})"

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key: str) -> Optional[Type]:
        """Get the registered class by key."""
        return self._module_dict.get(key)

    def register_module(self, name: Optional[str] = None, module: Optional[Type] = None) -> Union[Type, Callable]:
        """Register a module.

        Args:
            name (str, optional): The name of the module. If None, use the class name.
            module (Type, optional): The module class to be registered.
        """
        if module is not None:
            self._register_module(module=module, module_name=name)
            return module

        def _register(cls):
            self._register_module(module=cls, module_name=name)
            return cls

        return _register

    def _register_module(self, module: Type, module_name: Optional[str] = None):
        if not isinstance(module, type):
            raise TypeError(f"module must be a class, but got {type(module)}")

        if module_name is None:
            module_name = module.__name__

        if module_name in self._module_dict:
            raise KeyError(f"{module_name} is already registered in {self.name} registry")

        self._module_dict[module_name] = module

    def build(self, cfg: Dict[str, Any], **kwargs) -> Any:
        """Build an instance from config dict.

        Args:
            cfg (Dict[str, Any]): Config dict. Must contain 'type' key.
            **kwargs: Default arguments.

        Returns:
            Any: The constructed object.
        """
        if not isinstance(cfg, dict):
            raise TypeError(f"cfg must be a dict, but got {type(cfg)}")

        if 'type' not in cfg:
            raise KeyError(f"'type' must be in cfg")

        args = cfg.copy()
        obj_type = args.pop('type')

        if isinstance(obj_type, str):
            obj_cls = self.get(obj_type)
            if obj_cls is None:
                raise KeyError(f"{obj_type} is not in the {self.name} registry")
        elif isinstance(obj_type, type):
            obj_cls = obj_type
        else:
            raise TypeError(f"type must be str or class, but got {type(obj_type)}")

        # Merge kwargs into args (kwargs has higher priority if we want to override, 
        # or usually kwargs acts as defaults? 
        # In typical registries like mmcv, kwargs are default args if not present in cfg.
        # But here let's assume kwargs are extra arguments passed to constructor).
        # Let's simply update args with kwargs.
        args.update(kwargs)

        return obj_cls(**args)


# Create global registries
MODELS = Registry("model")
ATTACKERS = Registry("attacker")
DATASETS = Registry("dataset")
