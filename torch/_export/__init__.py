import torch
import weakref
from typing import Dict, Optional, Any, Callable
from torch.fx.experimental.symbolic_shapes import StrictMinMaxConstraint
from torch.fx._compatibility import compatibility
from torch.fx.passes import PassManager
from torch.fx.passes.infra.pass_base import PassResult
from torch.utils._sympy.value_ranges import ValueRanges
import sympy
from type_extensions import TypeAlias

PassType: TypeAlias = Callable[[torch.fx.GraphModule], Optional[PassResult]]
ExportGraphModule = torch.fx.GraphModule
# TODO(gmagogsfm): Rename this to EXPORT_METADATA and remove all references of EXIR_METADATA
EXIR_METADATA = "_exir_metadata_key"
Val = Any

# Note - [On Export Dynamic Dimension UX]
#
# After a lot of discussion, we have settled on a dynamic marking API
# for export that meets the following constraints:
# 1) Stateless
# 2) Safe for numerous .export calls within a single process
# 3) Simple to use
# 4) Can be extended to constraints easily
#
# While the underlying API is still torch._dynamo.mark_dynamic, we offer a higher
# level API that meets the constraints above.
#
# This API produces an object that is meant to be passed into torch._dynamo.export
# constraints field. See docs on torch._dynamo.export for more details.
#
# Note - The output type and structure here is NOT BC and NOT A CONTRACT, we reserve
# the right to change the output here at any time, and will do so as we extend the API.
#
# result = torch._dynamo.export(
#     my_model,
#     *sixtyfour_tensors,
#     constraints=[
#         # if you do only dynamic_dim, this is sugar for
#         # -Inf <= dynamic_dim(blah, 0) <= Inf; we don’t otherwise
#         # permit direct int->bool conversion
#         dynamic_dim(blah, 0),
#         # operator overloading because it makes it clear whether
#         # or not you’re inclusive-exclusive range or not
#         0 <= dynamic_dim(blah, 1) <= 100,
#         # NB: But we actually truncate ranges to be >= 2, because of
#         # 0/1 specialization
#     ]
# )
def dynamic_dim(t: torch.Tensor, index: int):
    from torch._dynamo.eval_frame import Constraint
    return Constraint(
        weakref.ref(t), id(t), index, StrictMinMaxConstraint(vr=ValueRanges(lower=2, upper=sympy.oo), warn_only=False)
    )

# MultiMethodExportedProgram represents an exported program that contains
# multiple methods, all as valid entry points to the program.
#
# Internally, each method is represented as a separate ExprotGraphModule.
# Methods (ExportGraphModule's) do not share anything with each other to
# ensure that each is self-contained. This is important because transformation
# passes can be local and do not need to concern themselves about other methods
# that exists on the same MultiMethodExportedProgram.
# TODO(gmagogsfm): Replace ExportedProgram with MultiMethodExportedProgram.

@compatibility(is_backward_compatible=False)
class MultiMethodExportedProgram:
    def __init__(self, gms: Dict[str, ExportGraphModule]):
        # TODO(gmagogsfm): Support merging use case where user started by creating
        # an empty MultiMethodExportedProgram and then start adding more
        # graph modules to it.
        assert (
            len(gms) > 0
        ), "Expected at least 1 graph module in MultiMethodExportedProgram"
        self._method_to_graph_module = gms

    # Get the default method, which is either the only method contained
    # in this MultiMethodExportedProgram or the method named `forward`.
    # TODO(gmagogsfm):Throw when there is only a single non-forward method in the program
    def _get_default_method(self):
        if len(self._method_to_graph_module) == 1:
            return next(iter(self._method_to_graph_module.values()))
        elif "forward" in self._method_to_graph_module:
            return self._method_to_graph_module["forward"]
        else:
            return None

    def save(self) -> None:
        # TODO(gmagogsfm): Implement.
        raise NotImplementedError()

    def load(self) -> None:
        # TODO(gmagogsfm): Implement.
        raise NotImplementedError()

    def find_method(self, name: str) -> Optional[torch.nn.Module]:
        return self._method_to_graph_module.get(name)

    def merge(self, other: "MultiMethodExportedProgram"):
        for method_name, gm in other.methods().items():
            assert (
                method_name not in self._method_to_graph_module
            ), f"There already is a method named {method_name} in this program"
            self._method_to_graph_module[method_name] = gm

    def transform(self, *passes: PassType) -> "MultiMethodExportedProgram":
        pm = PassManager(list(passes))

        def apply_passes(gm: ExportGraphModule) -> ExportGraphModule:
            transformed = pm(gm).graph_module
            assert transformed is not None
            transformed.meta.update(gm.meta)
            return transformed

        method_name_to_transformed_graph_modules = {
            method_name: apply_passes(gm)
            for method_name, gm in self._method_to_graph_module.items()
        }
        return MultiMethodExportedProgram(method_name_to_transformed_graph_modules)

    def methods(self) -> Dict[str, ExportGraphModule]:
        return self._method_to_graph_module

    def __call__(self, *args: Val, **kwargs: Val) -> Val:
        gm = self._get_default_method()

        assert (
            gm is not None
        ), """MultiMethodExportedProgram can not be called directly unless "
        "it only contains a single method or it contains a `forward` method. "
        "Please look up one of its methods first via "
        "`MultiMethodExportedProgram.find_method(method_name)`."""

        return gm(*args, **kwargs)

    def __repr__(self) -> str:
        # TODO(gmagogsfm): Implement.
        raise NotImplementedError()

    def __str__(self) -> str:
        # TODO(gmagogsfm): Implement a real one.
        return super().__str__()

    def access_property_of_default_method(self, property_name: str):
        default_module = self._get_default_method()
        assert (
            default_module is not None
        ), f"""Exported program contains more than one methods and none of them "
        "is named `forward`, it is impossible to identify the default method. "
        "please look up one of its methods first via `find_method(method_name)` "
        "to access property: {property_name}."""
        return getattr(default_module, property_name)

    @property
    def meta(self):
        return self.access_property_of_default_method("meta")

    @property
    def in_spec(self):
        return self.meta[EXIR_METADATA].in_spec

    @property
    def out_spec(self):
        return self.meta[EXIR_METADATA].out_spec

    @property
    def graph(self):
        return self.access_property_of_default_method("graph")

    @property
    def code(self):
        return self.access_property_of_default_method("code")

    @property
    def module(self):
        default_method = self._get_default_method()
        assert (
            default_method is not None
        ), """Exported program contains more than"
        " one methods and none of them is named `forward`,"
        " it is impossible to identify the default method "
        "to fetch GraphModule for."""
        return default_method

    # TODO(gmagogsfm): Implement custom __reduce__ to account for lost of
    # meta['val']
