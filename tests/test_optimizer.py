import copy
import io
import pickle
import runpy
from pathlib import Path

import pytest
import torch

import mndopt
from mndopt import MND


DTYPES = [torch.float64, torch.float32]


def test_public_api_and_default_options():
    assert mndopt.__all__ == ["MND"]
    assert MND.__bases__ == (torch.optim.Optimizer,)
    optimizer = MND([torch.nn.Parameter(torch.ones(2, 3))])
    assert optimizer.defaults == dict(lr=0.003, beta=0.925, eps=1e-9, weight_decay=0.0,
                                     matrix=None, adam_betas=(0.95, 0.95), adam_eps=1e-8)
    assert set(optimizer.state_dict()) == {"state", "param_groups"}


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("matrix", [True, False])
def test_checkpoint_exact_continuation(dtype, matrix):
    generator = torch.Generator().manual_seed(71)
    parameter = torch.nn.Parameter(torch.randn(2, 3, generator=generator, dtype=dtype))
    optimizer = MND([parameter], matrix=matrix, lr=0.001, weight_decay=0.03)
    gradients = [torch.randn(2, 3, generator=generator, dtype=dtype) for _ in range(8)]
    for gradient in gradients[:3]:
        parameter.grad = gradient.clone()
        optimizer.step()
    buffer = io.BytesIO()
    torch.save(optimizer.state_dict(), buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=True)
    other = torch.nn.Parameter(parameter.detach().clone())
    resumed = MND([other])
    resumed.load_state_dict(checkpoint)
    assert resumed.param_groups[0]["matrix"] == matrix
    for gradient in gradients[3:]:
        parameter.grad, other.grad = gradient.clone(), gradient.clone()
        optimizer.step()
        resumed.step()
        torch.testing.assert_close(parameter, other, rtol=0, atol=0)
        for name, value in optimizer.state[parameter].items():
            restored = resumed.state[other][name]
            if isinstance(value, torch.Tensor):
                assert restored.dtype == dtype
                assert restored.data_ptr() != value.data_ptr()
                torch.testing.assert_close(value, restored, rtol=0, atol=0)
            else:
                assert value == restored


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(2, 3), (3,)])
def test_missing_and_zero_gradients(dtype, shape):
    parameter = torch.nn.Parameter(torch.ones(shape, dtype=dtype))
    optimizer = MND([parameter], weight_decay=0.4)
    optimizer.step()
    assert parameter not in optimizer.state
    torch.testing.assert_close(parameter, torch.ones_like(parameter))
    optimizer.param_groups[0]["weight_decay"] = 0
    for _ in range(8):
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
    torch.testing.assert_close(parameter, torch.ones_like(parameter))
    assert optimizer.state[parameter]["step"] == 8
    for value in optimizer.state[parameter].values():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all()
    state_before = copy.deepcopy(optimizer.state[parameter])
    optimizer.zero_grad(set_to_none=True)
    optimizer.step()
    assert optimizer.state[parameter]["step"] == 8
    for name, value in state_before.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, optimizer.state[parameter][name], rtol=0, atol=0)


def test_closure_enables_gradients_and_returns_original_loss():
    parameter = torch.nn.Parameter(torch.ones(2, 3))
    optimizer = MND([parameter], lr=0.0001)
    losses = []

    def closure():
        assert torch.is_grad_enabled()
        optimizer.zero_grad()
        loss = parameter.square().sum()
        loss.backward()
        losses.append(loss)
        return loss

    with torch.no_grad():
        result = optimizer.step(closure)
    assert len(losses) == 1 and result is losses[0]
    assert result.item() == 6
    assert parameter.square().sum() < 6


def test_groups_match_separate_optimizers_and_add_group():
    parameters = [torch.nn.Parameter(torch.ones(2, 3, dtype=torch.float64)) for _ in range(3)]
    others = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
    options = [dict(lr=0.002, beta=0.8), dict(lr=0.001, beta=0.99, eps=1e-6),
               dict(matrix=False, adam_betas=(0.9, 0.999), weight_decay=0.1)]
    optimizer = MND([dict(params=[p], **o) for p, o in zip(parameters[:2], options[:2])])
    optimizer.add_param_group(dict(params=(p for p in parameters[2:]), **options[2]))
    controls = [MND([p], **o) for p, o in zip(others, options)]
    for index in range(5):
        for parameter, other in zip(parameters, others):
            parameter.grad = torch.full_like(parameter, (index - 2) / 10)
            other.grad = parameter.grad.clone()
        optimizer.step()
        for control in controls:
            control.step()
    for parameter, other in zip(parameters, others):
        torch.testing.assert_close(parameter, other, rtol=0, atol=0)


def test_auto_routing_and_zero_grad():
    parameters = [torch.nn.Parameter(torch.ones(shape)) for shape in [(), (3,), (2, 3), (2, 2, 2)]]
    optimizer = MND(parameters)
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    for parameter in parameters:
        assert optimizer.state[parameter]["matrix"] == (parameter.ndim == 2)
    optimizer.zero_grad(set_to_none=False)
    for parameter in parameters:
        assert parameter.grad is not None and (parameter.grad == 0).all()
    optimizer.zero_grad()
    assert all(parameter.grad is None for parameter in parameters)


@pytest.mark.parametrize("dtype", DTYPES)
def test_outer_autocast_preserves_exact_update(dtype):
    parameter = torch.nn.Parameter(torch.arange(6, dtype=dtype).reshape(2, 3) / 10)
    other = torch.nn.Parameter(parameter.detach().clone())
    optimizer, control = MND([parameter]), MND([other])
    generator = torch.Generator().manual_seed(8)
    for _ in range(4):
        gradient = torch.randn(2, 3, generator=generator, dtype=dtype)
        parameter.grad, other.grad = gradient.clone(), gradient.clone()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            optimizer.step()
        control.step()
    torch.testing.assert_close(parameter, other, rtol=0, atol=0)


def test_scheduler_and_optimizer_hooks():
    parameter = torch.nn.Parameter(torch.ones(2, 3))
    optimizer = MND([parameter], lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    events = []
    optimizer.register_step_pre_hook(lambda *args: events.append("pre"))
    optimizer.register_step_post_hook(lambda *args: events.append("post"))
    for _ in range(3):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.00125)
    assert events == ["pre", "post"] * 3


def test_load_hooks_can_transform_checkpoint():
    parameter = torch.nn.Parameter(torch.ones(2, 3))
    parameter.grad = torch.full_like(parameter, 0.3)
    optimizer = MND([parameter])
    optimizer.step()
    checkpoint = copy.deepcopy(optimizer.state_dict())
    other = torch.nn.Parameter(parameter.detach().clone())
    resumed = MND([other])
    original = checkpoint["state"][0]["momentum"].clone()

    def pre(optimizer, checkpoint):
        checkpoint = copy.deepcopy(checkpoint)
        checkpoint["state"][0]["momentum"] += 0.001234
        return checkpoint

    def post(optimizer):
        torch.testing.assert_close(optimizer.state[other]["momentum"], original + 0.001234,
                                   rtol=0, atol=0)
        assert optimizer.state[other]["momentum"].dtype == torch.float32

    resumed.register_load_state_dict_pre_hook(pre)
    resumed.register_load_state_dict_post_hook(post)
    resumed.load_state_dict(checkpoint)
    torch.testing.assert_close(checkpoint["state"][0]["momentum"], original, rtol=0, atol=0)


@pytest.mark.parametrize("method", [copy.deepcopy, lambda value: pickle.loads(pickle.dumps(value))])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("matrix", [True, False])
def test_optimizer_copy_and_pickle(method, dtype, matrix):
    parameter = torch.nn.Parameter(torch.ones(2, 3, dtype=dtype))
    parameter.grad = torch.ones_like(parameter)
    optimizer = MND([parameter], matrix=matrix)
    optimizer.step()
    restored = method(optimizer)
    other = restored.param_groups[0]["params"][0]
    parameter.grad, other.grad = torch.full_like(parameter, 0.2), torch.full_like(other, 0.2)
    optimizer.step()
    restored.step()
    torch.testing.assert_close(parameter, other, rtol=0, atol=0)


@pytest.mark.parametrize("key,value", [
    ("lr", -1), ("lr", float("inf")), ("lr", float("nan")),
    ("weight_decay", -1), ("weight_decay", float("inf")),
    ("eps", 0), ("eps", -1), ("eps", float("nan")),
    ("beta", 0), ("beta", 1), ("beta", -0.1), ("beta", float("nan")),
    ("matrix", 1), ("matrix", "yes"),
    ("adam_betas", (0.9,)), ("adam_betas", (-0.1, 0.9)),
    ("adam_betas", (0.9, 1)), ("adam_betas", (0.9, float("nan"))),
    ("adam_eps", 0), ("adam_eps", float("inf")),
])
@pytest.mark.parametrize("as_group", [False, True])
def test_invalid_options(key, value, as_group):
    parameter = torch.nn.Parameter(torch.ones(2, 3))
    options = {key: value}
    with pytest.raises(ValueError):
        if as_group:
            MND([dict(params=[parameter], **options)])
        else:
            MND([parameter], **options)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("matrix", [True, False])
def test_rejects_low_precision_parameters(dtype, matrix):
    parameter = torch.nn.Parameter(torch.ones(2, 3, dtype=dtype))
    parameter.grad = torch.ones_like(parameter)
    optimizer = MND([parameter], matrix=matrix)
    with pytest.raises(RuntimeError, match="float32 or float64"):
        optimizer.step()
    assert not optimizer.state


def test_invalid_added_group_does_not_change_optimizer():
    optimizer = MND([torch.nn.Parameter(torch.ones(2, 3))])
    with pytest.raises(ValueError):
        optimizer.add_param_group({"params": [torch.nn.Parameter(torch.ones(2, 3))], "lr": -1})
    assert len(optimizer.param_groups) == 1


def test_rejects_matrix_shape_and_routing_changes():
    with pytest.raises(ValueError, match="two-dimensional"):
        MND([torch.nn.Parameter(torch.ones(3))], matrix=True)
    parameter = torch.nn.Parameter(torch.ones(2, 3))
    parameter.grad = torch.ones_like(parameter)
    optimizer = MND([parameter])
    optimizer.step()
    optimizer.param_groups[0]["matrix"] = False
    before = parameter.detach().clone()
    with pytest.raises(ValueError, match="routing"):
        optimizer.step()
    torch.testing.assert_close(parameter, before, rtol=0, atol=0)
    assert optimizer.state[parameter]["step"] == 1


def test_rejects_sparse_complex_and_empty_gradients():
    parameter = torch.nn.Parameter(torch.ones(2, 3))
    parameter.grad = torch.sparse_coo_tensor(torch.tensor([[0], [1]]), torch.tensor([1.0]), (2, 3))
    with pytest.raises(RuntimeError, match="dense"):
        MND([parameter]).step()
    parameter = torch.nn.Parameter(torch.ones(2, 3, dtype=torch.complex64))
    parameter.grad = torch.ones_like(parameter)
    with pytest.raises(RuntimeError, match="real"):
        MND([parameter]).step()
    parameter = torch.nn.Parameter(torch.empty(0, 3))
    parameter.grad = torch.empty_like(parameter)
    with pytest.raises(ValueError, match="nonempty"):
        MND([parameter]).step()


def test_zero_learning_rate_updates_state_without_changing_parameters():
    parameter = torch.nn.Parameter(torch.ones(2, 3))
    parameter.grad = torch.ones_like(parameter)
    optimizer = MND([parameter], lr=0, weight_decay=0.3)
    optimizer.step()
    torch.testing.assert_close(parameter, torch.ones_like(parameter), rtol=0, atol=0)
    assert optimizer.state[parameter]["step"] == 1
    assert optimizer.state[parameter]["momentum"].abs().sum() > 0


def test_regression_converges_and_preserves_rng():
    example = Path(__file__).resolve().parents[1] / "examples" / "regression.py"
    run_regression = runpy.run_path(str(example))["run_regression"]

    before = torch.random.get_rng_state()
    losses = run_regression(steps=300)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert len(losses) == 301
    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < losses[0] * 0.02


def test_empty_checkpoint_keeps_state_lazy():
    parameter = torch.nn.Parameter(torch.ones(2, 3))
    optimizer = MND([parameter])
    optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    assert not optimizer.state


def test_checkpoint_group_mismatches_raise():
    parameter = torch.nn.Parameter(torch.ones(2, 3))
    optimizer = MND([parameter])
    checkpoint = copy.deepcopy(optimizer.state_dict())
    checkpoint["param_groups"].append(copy.deepcopy(checkpoint["param_groups"][0]))
    with pytest.raises(ValueError, match="number of parameter groups"):
        optimizer.load_state_dict(checkpoint)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", DTYPES)
def test_cuda_update_state_and_checkpoint(dtype):
    parameter = torch.nn.Parameter(torch.ones(2, 3, device="cuda", dtype=dtype))
    optimizer = MND([parameter])
    for _ in range(4):
        parameter.grad = torch.full_like(parameter, 0.2)
        optimizer.step()
    other = torch.nn.Parameter(parameter.detach().clone())
    resumed = MND([other])
    resumed.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    parameter.grad, other.grad = torch.full_like(parameter, 0.3), torch.full_like(other, 0.3)
    optimizer.step()
    resumed.step()
    torch.testing.assert_close(parameter, other, rtol=0, atol=0)
    assert torch.isfinite(parameter).all()
    for value in resumed.state[other].values():
        if isinstance(value, torch.Tensor):
            assert value.device == parameter.device
            assert value.dtype == dtype
