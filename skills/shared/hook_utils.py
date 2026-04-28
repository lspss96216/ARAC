"""hook_utils.py — v1.11.1

Base class and helpers for inject_modules hooks. Solves five independent
hook-related bugs that surface only after long training runs:

  #15 — Pickle failure on checkpoint save
        Closures cannot be pickled. Fix: hooks are top-level callable instances.

  #17 — AMP / val phase dtype mismatch
        Hook callable must dtype-cast input to its inner module's param dtype.

  #18 — `layer.forward = wrapper` bypasses _call_impl
        Use layer.register_forward_hook(callable) exclusively.

  v1.11.1 #5 — Hook sub-modules stay on CPU after trainer moves model to GPU
        PyTorch's model.to(device) only moves child modules tracked in
        _modules. Hook objects are forward-hook attributes, NOT child
        modules; their nn.Module sub-attributes (self.cbam etc.) get
        left on CPU when trainer moves the model to CUDA. Symptom:
          RuntimeError: Expected all tensors to be on the same device,
          but found at least two devices, cuda:0 and cpu!
        Fix: PicklableHook.__call__ now does lazy device sync on first
        call. Subclasses override forward() instead of __call__() to opt
        into auto sync. (Subclasses still overriding __call__ keep
        v1.7.7 - v1.11 behaviour for backward compat.)

  v1.11.1 #4 — reapply_on_rebuild silent hook loss on DetectionModel resolve
        ultralytics Trainer.get_model returns DetectionModel (m.model =
        Sequential), but inject_modules typically works against YOLO
        wrappers (m.model.model = ModuleList). User-written reapply
        callbacks did `for layer in m.model.model: ...` which raised
        AttributeError on DetectionModel. Previous WARNING-only handler
        swallowed silently → hooks unregistered → mAP == baseline.
        Fix: helper _get_layers(m) resolves both shapes; reapply_on_rebuild
        passes resolved `layers` (not `m`). WARNING → RuntimeError so
        failures are loud.
        BREAKING: callback signature changed from cb(m) to cb(layers).

Usage:

  class CBAMHook(PicklableHook):
      def __init__(self, channels: int):
          super().__init__()
          self.cbam = CBAM(channels)        # nn.Module owned by hook

      # v1.11.1: override forward() (not __call__) for auto device sync
      def forward(self, module, inputs, output):
          x = self._dtype_cast(output, self._param_dtype(self.cbam))
          y = self.cbam(x)
          return y.to(output.dtype)

  # Inside inject_modules:
  PicklableHook.attach(layer, CBAMHook, channels=256)

  # v1.11.1 reapply callback signature:
  def _reapply_my_hooks(layers):              # was: (m), now: (layers)
      PicklableHook.attach(layers[10], CBAMHook, channels=256)
  reapply_on_rebuild(model, _reapply_my_hooks)
"""

from __future__ import annotations
from typing import Any


def _get_layers(m: Any) -> Any:
    """v1.11.1 #4 — resolve a model object to its layer container.

    ultralytics' Trainer.get_model returns DetectionModel (m.model is the
    Sequential of layers). External code typically holds a YOLO wrapper
    (m.model.model is the ModuleList of layers). User-written reapply
    callbacks need to handle both. This helper does the resolution once.

    Returns:
      The ModuleList / Sequential of model layers, regardless of wrapper.

    Raises:
      RuntimeError if structure isn't recognised. Loud failure intentional —
      silent fallback was exactly what caused v1.11 #4's silent hook loss.
    """
    # Most-nested first to handle YOLO wrapper case correctly.
    if hasattr(m, "model") and hasattr(m.model, "model"):
        # YOLO wrapper: m.model is DetectionModel, m.model.model is ModuleList
        return m.model.model
    if hasattr(m, "model"):
        # DetectionModel: m.model is Sequential
        return m.model
    if hasattr(m, "__iter__") and hasattr(m, "__getitem__"):
        # Already a ModuleList or Sequential — pass through
        return m
    raise RuntimeError(
        f"_get_layers: unsupported model structure {type(m).__name__}; "
        f"expected YOLO wrapper (m.model.model), DetectionModel (m.model), "
        f"or directly a ModuleList/Sequential."
    )


class PicklableHook:
    """Base class for forward-hook callables that survive ckpt pickling.

    Subclasses MUST be defined at module top level (not nested in a function).

    v1.11.1 — two ways to subclass:

    1. Override forward() (recommended for new hooks):

         class MyHook(PicklableHook):
             def __init__(self, channels): ...
             def forward(self, module, inputs, output): ...

       Base class __call__ wraps forward() with auto device sync —
       internal nn.Module attributes get moved to output's device on
       first call. Solves v1.11.1 #5 silently.

    2. Override __call__ directly (legacy path, backward compat):

         class MyHook(PicklableHook):
             def __init__(self, ...): ...
             def __call__(self, module, inputs, output): ...

       In this path you're responsible for any device sync logic
       yourself. Same behaviour as v1.7.7 - v1.11.

    Helpers:
      - _param_dtype(submodule) → dtype of submodule's parameters
      - _dtype_cast(tensor, target_dtype) → cast if needed
      - attach(layer, cls, *args, **kwargs) → classmethod for register_forward_hook
    """

    # Empty __init__ so subclasses can call super().__init__() consistently.
    def __init__(self) -> None:
        # v1.11.1 #5 — track which device we last synced sub-modules to.
        # None means "never synced". On first __call__ we walk attributes
        # and move any nn.Module to output's device.
        self._synced_device: Any = None

    def __call__(self, module: Any, inputs: Any, output: Any) -> Any:
        """v1.11.1 #5 — auto device sync wrapper.

        If subclass overrode forward() (recommended): lazy-sync nn.Module
        attributes to output.device on first call, then dispatch to forward().

        If subclass overrode __call__ directly (legacy): this method is
        shadowed by the subclass version — behaviour identical to v1.7.7.
        """
        # forward() being default-NotImplementedError means subclass didn't
        # override it. If they ALSO didn't override __call__, that's a
        # broken subclass.
        if type(self).forward is PicklableHook.forward:
            raise NotImplementedError(
                f"{type(self).__name__} must override either forward() "
                f"(recommended) or __call__() (legacy)."
            )

        # Auto device sync. Cheap fast-path: same device as last call.
        target_device = output.device
        if self._synced_device != target_device:
            self._sync_submodules_to(target_device)
            self._synced_device = target_device

        return self.forward(module, inputs, output)

    def forward(self, module: Any, inputs: Any, output: Any) -> Any:
        """v1.11.1 — recommended subclass entry point. Override this for
        auto device sync. Default raises (sentinel for __call__ to detect)."""
        raise NotImplementedError(
            f"{type(self).__name__} must override forward(module, inputs, output)"
        )

    def _sync_submodules_to(self, target_device: Any) -> None:
        """v1.11.1 #5 — move every nn.Module attribute on this instance to
        target_device. Walks public attributes only (skip _-prefixed).

        Why this is needed: PyTorch's model.to(device) only moves child
        modules tracked in m._modules. Hook objects are stored as forward-
        hook attributes on layers, NOT as child modules. Their nn.Module
        sub-attributes (self.cbam, self.attn) need explicit movement.
        """
        try:
            import torch.nn as nn
        except ImportError:
            # No torch — testing environment. Skip sync.
            return

        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            try:
                attr = getattr(self, attr_name, None)
            except Exception:
                # Some attrs raise on access (e.g. property errors); skip.
                continue
            if isinstance(attr, nn.Module):
                # Avoid spurious work on parameter-less modules.
                has_params = next(attr.parameters(), None) is not None
                has_buffers = next(attr.buffers(), None) is not None
                if has_params or has_buffers:
                    setattr(self, attr_name, attr.to(target_device))

    @staticmethod
    def _param_dtype(submodule: Any):
        """Return dtype of submodule's first parameter, or None if no params."""
        try:
            for p in submodule.parameters():
                return p.dtype
        except (AttributeError, StopIteration):
            return None
        return None

    @staticmethod
    def _dtype_cast(tensor: Any, target_dtype: Any) -> Any:
        """Cast tensor to target_dtype iff it differs. No-op if target is None."""
        if target_dtype is None:
            return tensor
        if tensor.dtype != target_dtype:
            return tensor.to(target_dtype)
        return tensor

    @classmethod
    def attach(cls, layer: Any, *init_args: Any, **init_kwargs: Any):
        """Instantiate this hook and register on layer's forward output.

        Returns the RemovableHandle from register_forward_hook.
        Always uses register_forward_hook (NOT direct .forward = assignment).
        """
        hook = cls(*init_args, **init_kwargs)
        return layer.register_forward_hook(hook)


def reapply_on_rebuild(model: Any, reapply_fn: Any) -> None:
    """Register a callback that reapplies hooks after ultralytics' trainer
    rebuilds the model in setup_model.

    v1.7.7 #14 — ultralytics' Trainer.setup_model calls self.get_model()
    which constructs a fresh DetectionModel and loads weights. Any hooks
    registered on the ORIGINAL model from inject_modules() are bound to
    the OLD object and silently never fire on the rebuilt one. Fix:
    monkey-patch trainer.get_model so reapply_fn runs after the rebuild.

    v1.11.1 #4 — BREAKING CHANGE: callback signature changed.

      v1.7.7 - v1.11:   reapply_fn(rebuilt_nn_module)
      v1.11.1+:         reapply_fn(layers)

    The callback used to receive the raw rebuilt model object
    (DetectionModel), but most callers wrote `for layer in m.model.model:
    ...` expecting a YOLO wrapper. AttributeError ensued. The previous
    WARNING-only error handler swallowed this silently → "experiment
    looks like baseline because hooks never registered" silent corruption.

    Now `_get_layers(rebuilt)` is called inside reapply_on_rebuild, and
    the callback receives the resolved `layers` (ModuleList or Sequential)
    directly. Failures (unrecognised shape OR callback errors) raise
    RuntimeError loudly, never silent WARNING.

    Args:
      model:       the YOLO() instance returned by inject_modules
      reapply_fn:  callable(layers) -> None that re-attaches hooks to the
                   rebuilt model's layer container.

    Usage (v1.11.1):

      def _reapply_my_hooks(layers):
          PicklableHook.attach(layers[10], CBAMHook, channels=256)

      def inject_modules(model):
          layers = _get_layers(model)
          PicklableHook.attach(layers[10], CBAMHook, channels=256)
          reapply_on_rebuild(model, _reapply_my_hooks)
          return model
    """

    def _on_pretrain_routine_start(trainer: Any) -> None:
        original_get_model = trainer.get_model

        def patched_get_model(cfg: Any = None, weights: Any = None, verbose: bool = True):
            rebuilt = original_get_model(cfg=cfg, weights=weights, verbose=verbose)
            # v1.11.1 #4 — resolve the layer container, then call user's
            # callback with resolved layers (not the raw rebuilt object).
            try:
                layers = _get_layers(rebuilt)
            except RuntimeError as e:
                raise RuntimeError(
                    f"reapply_on_rebuild: cannot resolve layer container "
                    f"on rebuilt model ({type(rebuilt).__name__}). "
                    f"Original error: {e}. Hooks WILL be silently missing "
                    f"if we proceed; aborting instead."
                ) from e

            try:
                reapply_fn(layers)
            except Exception as e:
                raise RuntimeError(
                    f"reapply_on_rebuild: reapply_fn raised "
                    f"{type(e).__name__}: {e}. Hooks WILL be silently "
                    f"missing if we swallow this; aborting instead."
                ) from e
            return rebuilt

        trainer.get_model = patched_get_model

    model.add_callback("on_pretrain_routine_start", _on_pretrain_routine_start)
