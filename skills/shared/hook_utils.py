"""hook_utils.py — v1.7.7

Base class and helpers for inject_modules hooks. Solves three independent
hook-related bugs that surface only after long training runs:

  #15 — Pickle failure on checkpoint save
        Closures cannot be pickled; ultralytics torch.save(model) every
        epoch will fail with:
          Can't pickle local object 'apply_X.<locals>.make_hook.<locals>.hook'
        Symptom: train + val for epoch 1 succeed, then crash at ckpt write.
        Fix: hooks must be top-level callable instances, not closures.

  #17 — AMP / val phase dtype mismatch
        Training uses autocast (FP16); val phase doesn't (FP32). A hook
        with internal nn.Module gets FP32 input during val, FP16 input
        during train, and crashes with:
          RuntimeError: Input type (HalfTensor) and weight type (FloatTensor)
                        should be the same
        Symptom: train all batches succeed, val batch 0 crashes.
        Fix: hook callable must dtype-cast input to match its inner
        module's parameter dtype.

  #18 — `layer.forward = wrapper` bypasses _call_impl
        Direct method override skips PyTorch's hook dispatch system and
        breaks during fused / val paths in some ultralytics codepaths.
        Fix: use `layer.register_forward_hook(callable)` exclusively.

Usage:

  class CBAMHook(PicklableHook):
      def __init__(self, channels: int):
          super().__init__()
          self.cbam = CBAM(channels)        # nn.Module owned by hook

      def __call__(self, module, inputs, output):
          # _dtype_cast handles AMP/val mismatch (#17)
          x = self._dtype_cast(output, self._param_dtype(self.cbam))
          y = self.cbam(x)
          # Cast back to caller's dtype so downstream layers see what they expect
          return y.to(output.dtype)

  # Inside inject_modules:
  PicklableHook.attach(layer, CBAMHook, channels=256)   # uses register_forward_hook (#18)
"""

from __future__ import annotations
from typing import Any


class PicklableHook:
    """Base class for forward-hook callables that survive ckpt pickling.

    Subclasses MUST be defined at module top level (not nested inside a
    function). Override __call__(self, module, inputs, output).

    Helpers:
      - _param_dtype(submodule) → dtype of submodule's parameters (or None)
      - _dtype_cast(tensor, target_dtype) → cast if needed, else tensor as-is
      - attach(layer, cls, *args, **kwargs) → classmethod for register_forward_hook
    """

    # Empty __init__ so subclasses can call super().__init__() consistently.
    def __init__(self) -> None:
        pass

    def __call__(self, module: Any, inputs: Any, output: Any) -> Any:
        raise NotImplementedError(
            f"{type(self).__name__} must override __call__(module, inputs, output)"
        )

    @staticmethod
    def _param_dtype(submodule: Any):
        """Return the dtype of submodule's first parameter, or None if no params.

        Used to align hook input dtype with hook's internal nn.Module weights.
        Submodule may be a torch.nn.Module or any object exposing .parameters().
        """
        try:
            for p in submodule.parameters():
                return p.dtype
        except (AttributeError, StopIteration):
            return None
        return None

    @staticmethod
    def _dtype_cast(tensor: Any, target_dtype: Any) -> Any:
        """Cast tensor to target_dtype iff it differs. No-op if target is None.

        Cheap fast-path for the common case (matching dtype) — costs one
        attribute read.
        """
        if target_dtype is None:
            return tensor
        if tensor.dtype != target_dtype:
            return tensor.to(target_dtype)
        return tensor

    @classmethod
    def attach(cls, layer: Any, *init_args: Any, **init_kwargs: Any):
        """Instantiate this hook and register on layer's forward output.

        Returns the RemovableHandle from register_forward_hook so the caller
        can remove it later if needed.

        Always uses register_forward_hook (NOT direct .forward = assignment).
        """
        hook = cls(*init_args, **init_kwargs)
        return layer.register_forward_hook(hook)


def reapply_on_rebuild(model: Any, reapply_fn: Any) -> None:
    """Register a callback that reapplies hooks after ultralytics' trainer
    rebuilds the model in setup_model.

    v1.7.7 #14 — ultralytics' Trainer.setup_model calls self.get_model() which
    constructs a *fresh* DetectionModel and loads weights into it. Any hooks
    registered on the original model from inject_modules() are bound to the
    OLD object and silently never fire on the rebuilt one. The fix is to
    monkey-patch trainer.get_model so reapply_fn(new_model) runs after the
    rebuild.

    Args:
      model:       the YOLO() instance returned by inject_modules
      reapply_fn:  callable(rebuilt_nn_module) -> None that re-attaches all
                   hooks to the rebuilt model. Typically a thin wrapper around
                   the same logic inject_modules used.

    Usage:
      def _reapply_my_hooks(m):
          PicklableHook.attach(m.model[10], CBAMHook, channels=256)

      def inject_modules(model):
          PicklableHook.attach(model.model.model[10], CBAMHook, channels=256)
          reapply_on_rebuild(model, _reapply_my_hooks)
          return model
    """

    def _on_pretrain_routine_start(trainer: Any) -> None:
        original_get_model = trainer.get_model

        def patched_get_model(cfg: Any = None, weights: Any = None, verbose: bool = True):
            rebuilt = original_get_model(cfg=cfg, weights=weights, verbose=verbose)
            try:
                reapply_fn(rebuilt)
            except Exception as e:
                # Log but don't crash training — agent should see the silent
                # hook-loss symptom (mAP == baseline) and inspect, rather than
                # masking the rebuild path entirely.
                import sys
                print(
                    f"[reapply_on_rebuild] WARNING: reapply_fn raised "
                    f"{type(e).__name__}: {e}. Hooks may be missing on the "
                    f"rebuilt model.",
                    file=sys.stderr,
                )
            return rebuilt

        trainer.get_model = patched_get_model

    model.add_callback("on_pretrain_routine_start", _on_pretrain_routine_start)
