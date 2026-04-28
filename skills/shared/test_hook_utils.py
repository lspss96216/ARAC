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


# v1.11.1 #5 — tensor stand-in with .device for auto sync tests
class FakeTensorWithDevice:
    def __init__(self, dtype, device):
        self.dtype = dtype
        self.device = device
    def to(self, target):
        return FakeTensorWithDevice(target if isinstance(target, str) else self.dtype, self.device)
    def __repr__(self):
        return f"FakeTensorWithDevice({self.dtype}, {self.device})"


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
    """v1.7.7 #14 + v1.11.1 #4 — callback gets resolved layers, not raw model.
    trainer.get_model is monkey-patched such that it resolves the rebuilt
    model's layer container and passes that to reapply_fn."""
    reapply_calls = []
    def my_reapply(layers):                   # v1.11.1 signature: (layers)
        reapply_calls.append(layers)

    # Rebuilt model has DetectionModel shape: m.model = layer container
    fake_layer_container = ["layer0", "layer1", "layer2"]
    class FakeRebuilt:
        def __init__(self):
            self.model = fake_layer_container

    class FakeTrainer:
        def get_model(self, cfg=None, weights=None, verbose=True):
            return FakeRebuilt()

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
    # Returned object is the rebuilt model itself
    assert isinstance(got, FakeRebuilt)
    # Callback received the RESOLVED layers, not the raw rebuilt model
    assert reapply_calls == [fake_layer_container], reapply_calls
    print("✓ test_reapply_callback_patches_get_model")


def test_reapply_callback_resolves_yolo_wrapper_shape():
    """v1.11.1 #4 — when rebuilt is a YOLO wrapper (m.model.model = layers),
    callback still gets the inner ModuleList, not m.model."""
    reapply_calls = []
    def my_reapply(layers):
        reapply_calls.append(layers)

    fake_layer_list = ["L0", "L1"]
    class InnerDetectionModel:
        def __init__(self):
            self.model = fake_layer_list   # ModuleList lives here
    class FakeRebuiltYOLO:
        def __init__(self):
            self.model = InnerDetectionModel()   # YOLO wrapper

    class FakeTrainer:
        def get_model(self, cfg=None, weights=None, verbose=True):
            return FakeRebuiltYOLO()

    class FakeYOLO:
        def __init__(self): self.callbacks = {}
        def add_callback(self, n, f): self.callbacks[n] = f

    yolo = FakeYOLO()
    hu.reapply_on_rebuild(yolo, my_reapply)
    trainer = FakeTrainer()
    yolo.callbacks["on_pretrain_routine_start"](trainer)
    trainer.get_model()

    # Callback got the INNER ModuleList, not the outer DetectionModel
    assert reapply_calls == [fake_layer_list], reapply_calls
    print("✓ test_reapply_callback_resolves_yolo_wrapper_shape")


def test_reapply_raises_on_reapply_error():
    """v1.11.1 #4 — if reapply_fn raises, RuntimeError propagates loudly.
    v1.7.7-v1.11 swallowed silently as WARNING; that caused hooks to be
    missing on rebuilt model with no signal beyond mAP == baseline.
    Loud failure now."""
    def bad_reapply(layers):
        raise ValueError("simulated reapply failure")

    class FakeRebuilt:
        def __init__(self): self.model = ["L0"]
    class FakeTrainer:
        def get_model(self, cfg=None, weights=None, verbose=True):
            return FakeRebuilt()
    class FakeYOLO:
        def __init__(self): self.callbacks = {}
        def add_callback(self, n, f): self.callbacks[n] = f

    yolo = FakeYOLO()
    hu.reapply_on_rebuild(yolo, bad_reapply)
    trainer = FakeTrainer()
    yolo.callbacks["on_pretrain_routine_start"](trainer)

    try:
        trainer.get_model()
    except RuntimeError as e:
        assert "reapply_fn raised" in str(e)
        assert "ValueError" in str(e)
        assert "simulated reapply failure" in str(e)
        print("✓ test_reapply_raises_on_reapply_error")
        return
    raise AssertionError("expected RuntimeError, got none")


def test_reapply_raises_on_unrecognised_model_shape():
    """v1.11.1 #4 — if rebuilt model has neither m.model nor iterable
    structure, _get_layers raises and reapply_on_rebuild propagates."""
    reapply_calls = []
    def my_reapply(layers):
        reapply_calls.append(layers)

    class FakeTrainer:
        def get_model(self, cfg=None, weights=None, verbose=True):
            return object()   # opaque, no .model attr

    class FakeYOLO:
        def __init__(self): self.callbacks = {}
        def add_callback(self, n, f): self.callbacks[n] = f

    yolo = FakeYOLO()
    hu.reapply_on_rebuild(yolo, my_reapply)
    trainer = FakeTrainer()
    yolo.callbacks["on_pretrain_routine_start"](trainer)

    try:
        trainer.get_model()
    except RuntimeError as e:
        assert "cannot resolve layer container" in str(e)
        assert reapply_calls == []   # callback never invoked
        print("✓ test_reapply_raises_on_unrecognised_model_shape")
        return
    raise AssertionError("expected RuntimeError, got none")


# ─── v1.11.1 #5 auto device sync tests ─────────────────────────────────────

class _SampleHookForwardPath(hu.PicklableHook):
    """v1.11.1 — subclass overrides forward() (recommended path)."""
    def __init__(self, channels=64):
        super().__init__()
        self.channels = channels
        self.calls = []
    def forward(self, module, inputs, output):
        self.calls.append(("forward", output.dtype if hasattr(output, "dtype") else None))
        return output


def test_forward_path_dispatches_through_call():
    """v1.11.1 #5 — subclass that overrides forward() (not __call__)
    gets dispatched through __call__ wrapper."""
    h = _SampleHookForwardPath(channels=128)
    out = FakeTensorWithDevice("float32", "cpu")
    result = h(None, None, out)
    assert result is out
    assert h.calls == [("forward", "float32")]
    print("✓ test_forward_path_dispatches_through_call")


def test_no_override_raises_on_call():
    """v1.11.1 — subclass that overrides NEITHER forward() NOR __call__()
    raises NotImplementedError (sentinel for broken subclass)."""
    class _BothMissing(hu.PicklableHook):
        pass
    out = FakeTensorWithDevice("float32", "cpu")
    try:
        _BothMissing()(None, None, out)
    except NotImplementedError as e:
        assert "_BothMissing" in str(e)
        # v1.11.1 message mentions both options
        assert "forward()" in str(e) or "__call__()" in str(e)
        print("✓ test_no_override_raises_on_call")
        return
    raise AssertionError("expected NotImplementedError")


def test_sync_submodules_skipped_when_no_torch():
    """v1.11.1 #5 — _sync_submodules_to is a no-op when torch unavailable.
    In real env (torch installed) the iteration runs; here we only verify
    it doesn't crash."""
    h = _SampleHookForwardPath()
    # Calling _sync_submodules_to directly should not raise even without torch
    h._sync_submodules_to("cuda:0")
    print("✓ test_sync_submodules_skipped_when_no_torch")


def test_synced_device_caches_after_first_call():
    """v1.11.1 #5 — _synced_device records the device on first __call__,
    avoids re-walk on subsequent calls with same device."""
    h = _SampleHookForwardPath()
    assert h._synced_device is None  # initial state
    out = FakeTensorWithDevice("float32", "cuda:0")
    h(None, None, out)
    assert h._synced_device == "cuda:0"
    # Second call same device — no change
    h(None, None, out)
    assert h._synced_device == "cuda:0"
    # Different device — _synced_device updates
    out2 = FakeTensorWithDevice("float32", "cuda:1")
    h(None, None, out2)
    assert h._synced_device == "cuda:1"
    print("✓ test_synced_device_caches_after_first_call")


# ─── v1.11.1 #4 _get_layers tests ──────────────────────────────────────────


def test_get_layers_resolves_yolo_wrapper():
    """v1.11.1 #4 — _get_layers returns m.model.model when both exist."""
    inner = ["L0", "L1"]
    class Inner: pass
    inner_obj = Inner(); inner_obj.model = inner
    class Outer: pass
    outer = Outer(); outer.model = inner_obj
    result = hu._get_layers(outer)
    assert result is inner
    print("✓ test_get_layers_resolves_yolo_wrapper")


def test_get_layers_resolves_detection_model():
    """v1.11.1 #4 — _get_layers returns m.model when m.model.model doesn't exist."""
    layers = ["L0", "L1", "L2"]
    class DetModel: pass
    m = DetModel(); m.model = layers
    result = hu._get_layers(m)
    assert result is layers
    print("✓ test_get_layers_resolves_detection_model")


def test_get_layers_passthrough_for_iterable():
    """v1.11.1 #4 — _get_layers returns m as-is if it's already a sequence."""
    layers = ["L0", "L1"]
    result = hu._get_layers(layers)
    assert result is layers
    print("✓ test_get_layers_passthrough_for_iterable")


def test_get_layers_raises_for_unrecognised_shape():
    """v1.11.1 #4 — opaque object with no .model and no iterable interface raises."""
    try:
        hu._get_layers(object())
    except RuntimeError as e:
        assert "unsupported model structure" in str(e)
        print("✓ test_get_layers_raises_for_unrecognised_shape")
        return
    raise AssertionError("expected RuntimeError")


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
    # v1.11.1 #4 — new tests for (layers) callback signature + loud failure
    test_reapply_callback_resolves_yolo_wrapper_shape,
    test_reapply_raises_on_reapply_error,
    test_reapply_raises_on_unrecognised_model_shape,
    # v1.11.1 #5 — auto device sync + forward() path
    test_forward_path_dispatches_through_call,
    test_no_override_raises_on_call,
    test_sync_submodules_skipped_when_no_torch,
    test_synced_device_caches_after_first_call,
    # v1.11.1 #4 — _get_layers helper standalone tests
    test_get_layers_resolves_yolo_wrapper,
    test_get_layers_resolves_detection_model,
    test_get_layers_passthrough_for_iterable,
    test_get_layers_raises_for_unrecognised_shape,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
    print(f"\nall {len(TESTS)} tests passed")
