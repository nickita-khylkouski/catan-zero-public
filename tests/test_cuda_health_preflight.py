from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools import cuda_health_preflight as preflight


class _FakeTensor:
    def __init__(self, value: float) -> None:
        self.value = value

    def add_(self, value: float) -> "_FakeTensor":
        self.value += value
        return self

    def item(self) -> float:
        return self.value


class _FakeCuda:
    def __init__(self, *, available: bool = True, devices: int = 2) -> None:
        self.available = available
        self.devices = devices
        self.selected: list[int] = []
        self.synchronized: list[int] = []

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return self.devices

    def set_device(self, index: int) -> None:
        self.selected.append(index)

    def synchronize(self, index: int) -> None:
        self.synchronized.append(index)


class _FakeDistributed:
    ReduceOp = SimpleNamespace(SUM="sum")

    def __init__(self, *, collective_sum: float = 3.0) -> None:
        self.collective_sum = collective_sum
        self.initialized: list[tuple[str, object]] = []
        self.destroyed = False

    def init_process_group(self, *, backend: str, timeout: object) -> None:
        self.initialized.append((backend, timeout))

    def all_reduce(self, value: _FakeTensor, *, op: object) -> None:
        assert op == self.ReduceOp.SUM
        value.value = self.collective_sum

    def destroy_process_group(self) -> None:
        self.destroyed = True


class _FakeTorch:
    float32 = "float32"

    def __init__(
        self,
        *,
        available: bool = True,
        devices: int = 2,
        collective_sum: float = 3.0,
    ) -> None:
        self.cuda = _FakeCuda(available=available, devices=devices)
        self.distributed = _FakeDistributed(collective_sum=collective_sum)

    def ones(self, _size: int, *, dtype: object, device: str) -> _FakeTensor:
        assert dtype == self.float32
        assert device.startswith("cuda:")
        return _FakeTensor(1.0)

    def tensor(self, values: list[float], *, dtype: object, device: str) -> _FakeTensor:
        assert dtype == self.float32
        assert device.startswith("cuda:")
        return _FakeTensor(values[0])


def test_allocation_probe_touches_and_synchronizes_every_visible_device() -> None:
    torch = _FakeTorch(devices=2)

    preflight.check_allocations(torch, expected_devices=2)

    assert torch.cuda.selected == [0, 1]
    assert torch.cuda.synchronized == [0, 1]


@pytest.mark.parametrize(
    ("torch", "expected", "message"),
    [
        (_FakeTorch(available=False), 2, "is_available"),
        (_FakeTorch(devices=1), 2, "expected 2 visible"),
    ],
)
def test_allocation_probe_fails_closed(
    torch: _FakeTorch, expected: int, message: str
) -> None:
    with pytest.raises(preflight.PreflightError, match=message):
        preflight.check_allocations(torch, expected_devices=expected)


def test_collective_probe_allocates_local_rank_and_reduces() -> None:
    torch = _FakeTorch(devices=2, collective_sum=3.0)

    preflight.check_nccl_collective(
        torch,
        expected_devices=2,
        environ={"LOCAL_RANK": "1", "RANK": "1", "WORLD_SIZE": "2"},
    )

    assert torch.cuda.selected == [1]
    assert torch.cuda.synchronized == [1, 1]
    assert torch.distributed.initialized[0][0] == "nccl"
    assert torch.distributed.destroyed is True


def test_collective_probe_rejects_world_size_mismatch_before_cuda() -> None:
    torch = _FakeTorch(devices=2)

    with pytest.raises(preflight.PreflightError, match="world size 1"):
        preflight.check_nccl_collective(
            torch,
            expected_devices=2,
            environ={"LOCAL_RANK": "0", "RANK": "0", "WORLD_SIZE": "1"},
        )

    assert torch.cuda.selected == []
