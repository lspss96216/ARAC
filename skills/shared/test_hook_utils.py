"""test_hook_utils.py — v1.7.7

Pure-python tests for hook_utils. Avoids depending on torch so it runs in
the same fast suite as test_modules_md / test_weight_transfer.

Coverage:
  - PicklableHook subclass instantiates
  - PicklableHook subclass is picklable (#15)
  - _dtype_cast no-ops on matching dtype, casts on mismatch (#17)
  - _param_dtype returns first parameter's dtype, None if no params
  - attach() uses register_forward_hook (#18) with correct args
  - reapply_on_rebuild registers callback that monkey-patches get_model (#14)
"""
import pickle
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import hook_utils as hu


# Stand-in for a torch.Tensor with just .dtype + .to() — keeps tests
# torch-free.
class FakeTensor:
    def __init__(self, dtype):
        self.dtype = dtype
    def to(self, target_dtype):
        return FakeTensor(target_dtype)
    def __eq__(self, other):
        return isinstance(other, FakeTensor) and self.dtype == other.dtype
    def __repr__(self):
        return f"FakeTensor({self.dtype})"


# Stand-in for an nn.Module with .parameters() that yields fake params with .dtype
class FakeParam:
    def __init__(self, dtype): self.dtype = dtype


class FakeModule:
    def __init__(self, dtype=None):
        self._dtype = dtype
    def parameters(self):
        if self._dtype is None:
            return iter([])
        return iter([FakeParam(self._dtype)])


# Stand-in for a layer accepting register_forward_hook
class FakeLayer:
    def __init__(self):
        self.registered = []
    def register_forward_hook(self, hook):
        self.registered.append(hook)
        return ("handle", id(hook))


# ── tests ────────────────────────────────────────────────────────────────


class _SampleHook(hu.PicklableHook):
    """Top-level subclass — must be defined at module level for pickling."""
    def __init__(self, channels=64):
        super().__init__()
        self.channels = channels
    def __call__(self, module, inputs, output):
        return output


def test_subclass_instantiates():
    h = _SampleHook(channels=128)
    assert h.channels == 128
    print("✓ test_subclass_instantiates")


def test_subclass_is_picklable():
    """Bug #15 — closures fail to pickle; top-level class must succeed."""
    h = _SampleHook(channels=64)
    blob = pickle.dumps(h)
    h2 = pickle.loads(blob)
    assert isinstance(h2, _SampleHook)
    assert h2.channels == 64
    print("✓ test_subclass_is_picklable")


def test_base_class_raises_on_call():
    """PicklableHook itself raises NotImplementedError on __call__."""
    class _Bare(hu.PicklableHook):
        pass
    try:
        _Bare()(None, None, None)
    except NotImplementedError as e:
        assert "_Bare" in str(e), str(e)
        print("✓ test_base_class_raises_on_call")
        return
    raise AssertionError("expected NotImplementedError")


def test_dtype_cast_noop_on_match():
    """Bug #17 fast-path — no copy when dtypes match."""
    t = FakeTensor("float32")
    out = hu.PicklableHook._dtype_cast(t, "float32")
    assert out is t   # identity — exact same object
    print("✓ test_dtype_cast_noop_on_match")


def test_dtype_cast_converts_on_mismatch():
    """Bug #17 — actual cast when dtypes differ (val phase getting FP16)."""
    t = FakeTensor("float16")
    out = hu.PicklableHook._dtype_cast(t, "float32")
    assert out.dtype == "float32"
    assert out is not t
    print("✓ test_dtype_cast_converts_on_mismatch")


def test_dtype_cast_target_none_is_noop():
    """When target_dtype is None (no params in submodule), pass through."""
    t = FakeTensor("float32")
    out = hu.PicklableHook._dtype_cast(t, None)
    assert out is t
    print("✓ test_dtype_cast_target_none_is_noop")


def test_param_dtype_returns_first():
    m = FakeModule(dtype="float32")
    assert hu.PicklableHook._param_dtype(m) == "float32"
    print("✓ test_param_dtype_returns_first")


def test_param_dtype_returns_none_when_no_params():
    m = FakeModule(dtype=None)
    assert hu.PicklableHook._param_dtype(m) is None
    print("✓ test_param_dtype_returns_none_when_no_params")


def test_param_dtype_returns_none_on_attribute_error():
    """Object without .parameters() — return None, don't crash."""
    class NoParams: pass
    assert hu.PicklableHook._param_dtype(NoParams()) is None
    print("✓ test_param_dtype_returns_none_on_attribute_error")


def test_attach_uses_register_forward_hook():
    """Bug #18 — attach() must use register_forward_hook, never .forward = ."""
    layer = FakeLayer()
    handle = _SampleHook.attach(layer, channels=32)
    assert len(layer.registered) == 1, "should have called register_forward_hook once"
    assert isinstance(layer.registered[0], _SampleHook)
    assert layer.registered[0].channels == 32
    assert handle[0] == "handle"
    print("✓ test_attach_uses_register_forward_hook")


def test_reapply_on_rebuild_registers_callback():
    """Bug #14 — reapply_on_rebuild registers on_pretrain_routine_start."""
    callbacks_added = []
    class FakeYOLO:
        def add_callback(self, name, fn):
            callbacks_added.append((name, fn))
    model = FakeYOLO()
    def my_reapply(m): pass
    hu.reapply_on_rebuild(model, my_reapply)
    assert len(callbacks_added) == 1
    assert callbacks_added[0][0] == "on_pretrain_routine_start"
    assert callable(callbacks_added[0][1])
    print("✓ test_reapply_on_rebuild_registers_callback")


def test_reapply_callback_patches_get_model():
    """Bug #14 — when callback fires, trainer.get_model gets monkey-patched
    such that the patched version calls reapply_fn on the rebuilt model."""
    reapply_calls = []
    def my_reapply(m):
        reapply_calls.append(m)

    rebuilt_model = object()   # sentinel for "the model trainer rebuilt"

    class FakeTrainer:
        def get_model(self, cfg=None, weights=None, verbose=True):
            return rebuilt_model

    class FakeYOLO:
        def __init__(self):
            self.callbacks = {}
        def add_callback(self, name, fn):
            self.callbacks[name] = fn

    yolo = FakeYOLO()
    hu.reapply_on_rebuild(yolo, my_reapply)

    # Simulate ultralytics firing the callback during training start
    trainer = FakeTrainer()
    yolo.callbacks["on_pretrain_routine_start"](trainer)

    # trainer.get_model is now patched. Call it as ultralytics would.
    got = trainer.get_model(cfg=None, weights="x.pt", verbose=False)
    assert got is rebuilt_model
    assert reapply_calls == [rebuilt_model], reapply_calls
    print("✓ test_reapply_callback_patches_get_model")


def test_reapply_does_not_crash_training_on_reapply_error():
    """If reapply_fn raises, training continues — symptom (mAP == baseline)
    is observable to the agent later. Crashing here would mask the bug."""
    def bad_reapply(m):
        raise RuntimeError("simulated reapply failure")

    rebuilt = object()
    class FakeTrainer:
        def get_model(self, cfg=None, weights=None, verbose=True):
            return rebuilt

    class FakeYOLO:
        def __init__(self): self.callbacks = {}
        def add_callback(self, name, fn): self.callbacks[name] = fn

    yolo = FakeYOLO()
    hu.reapply_on_rebuild(yolo, bad_reapply)
    trainer = FakeTrainer()
    yolo.callbacks["on_pretrain_routine_start"](trainer)
    # Patched get_model must still return the rebuilt model even when
    # reapply raises — training proceeds, agent will see baseline mAP.
    got = trainer.get_model()
    assert got is rebuilt
    print("✓ test_reapply_does_not_crash_training_on_reapply_error")


TESTS = [
    test_subclass_instantiates,
    test_subclass_is_picklable,
    test_base_class_raises_on_call,
    test_dtype_cast_noop_on_match,
    test_dtype_cast_converts_on_mismatch,
    test_dtype_cast_target_none_is_noop,
    test_param_dtype_returns_first,
    test_param_dtype_returns_none_when_no_params,
    test_param_dtype_returns_none_on_attribute_error,
    test_attach_uses_register_forward_hook,
    test_reapply_on_rebuild_registers_callback,
    test_reapply_callback_patches_get_model,
    test_reapply_does_not_crash_training_on_reapply_error,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
    print(f"\nall {len(TESTS)} tests passed")
