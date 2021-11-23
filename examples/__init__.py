class RayException(Exception):
    """An exception raised if ray package is required but not available."""

    @classmethod
    def required_to(cls, thing):
        return cls(
            f"""Ray Package is required to simulate {thing}.
               You may not have installed the [train] or [test] dependencies required to run the ray dependent example.
               Install them first using the command `pip install -e .[train, test]` at the source directory to install the package ray[rllib]==1.0.1.post1"""
        )


from pathlib import Path

from smarts.core.utils import import_utils

from . import argument_parser as argument_parser

import_utils.import_module_from_file(
    "examples", Path(__file__).parents[1] / "__init__.py"
)
