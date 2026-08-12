"""
Utility functions and classes for efficient transformer model management in Nunchaku.
"""

import copy

import torch
from torch import nn

from ..utils import copy_params_into as _copy_params_into

TensorPair = tuple[torch.Tensor, torch.Tensor]
ModulePair = tuple[nn.Module, nn.Module]
PackedParameterSlabs = dict[torch.dtype, torch.Tensor]
PackedCopyPlan = tuple[tuple[TensorPair, ...], tuple[TensorPair, ...], tuple[ModulePair, ...]]


def copy_params_into(
    src: nn.Module,
    dst: nn.Module,
    non_blocking: bool = True,
    copy_plan: PackedCopyPlan | None = None,
):
    """
    Copy a module using either the generic parameter walk or a packed slab plan.

    ``CPUOffloadManager`` uses the packed path to collapse many per-parameter
    H2D copies into a small number of dtype-grouped slab copies while keeping
    this helper name stable for profiling hooks and existing imports.
    """
    if copy_plan is None:
        _copy_params_into(src, dst, non_blocking=non_blocking)
        return

    slab_pairs, buffer_pairs, module_pairs = copy_plan
    with torch.no_grad():
        for src_slab, dst_slab in slab_pairs:
            dst_slab.copy_(src_slab, non_blocking=non_blocking)
        for src_buffer, dst_buffer in buffer_pairs:
            dst_buffer.copy_(src_buffer, non_blocking=non_blocking)
        for src_module, dst_module in module_pairs:
            dst_module.wtscale = src_module.wtscale


def fuse_linears(linears: list[nn.Linear]) -> nn.Linear:
    """
    Fuse a list of nn.Linear layers into a single nn.Linear with concatenated output features.

    Parameters
    ----------
    linears : list of nn.Linear
        List of linear layers to fuse. All must have the same input feature dimension.

    Returns
    -------
    fused : nn.Linear
        A new linear layer with concatenated output features and the same input features.

    Raises
    ------
    AssertionError
        If the input feature dimensions do not match.

    Notes
    -----
    The fused layer does not copy weights or biases from the input layers.
    """
    assert len(linears) > 0
    if len(linears) == 1:
        return linears[0]
    else:
        assert all(linear.in_features == linears[0].in_features for linear in linears)
        out_features = sum(linear.out_features for linear in linears)
        bias = all(linear.bias is not None for linear in linears)
        return nn.Linear(
            linears[0].in_features,
            out_features,
            bias=bias,
            dtype=linears[0].weight.dtype,
            device=linears[0].weight.device,
        )


class CPUOffloadManager:
    """
    Manager for per-transformer-block CPU offloading with asynchronous memory operations using a Ping-Pong buffer strategy.

    This class enables memory-efficient inference or training by keeping only a subset
    of transformer blocks on GPU, offloading the rest to CPU, and preloading blocks as needed.

    Parameters
    ----------
    blocks : list of nn.Module
        List of transformer blocks to manage.
    device : str or torch.device, optional
        Target CUDA device for GPU operations. Default is "cuda".
    use_pin_memory : bool, optional
        Whether to use pinned memory for faster CPU-to-GPU transfers. Default is True.
    on_gpu_modules : list of nn.Module, optional
        Additional modules to keep on GPU at all times. Default is [].
    num_blocks_on_gpu : int, optional
        Number of blocks to keep on GPU simultaneously. Must be > 0. Default is 1.
    empty_cache_freq : int, optional
        Frequency (in forward passes) to call torch.cuda.empty_cache(). Default is 0 (never).

    Attributes
    ----------
    blocks : list of nn.Module
        The managed transformer blocks.
    buffer_blocks : list of nn.Module
        Buffers for preloading blocks onto GPU.
    device : torch.device
        The current CUDA device.
    current_block_idx : int
        Index of the current block on GPU.
    forward_counter : int
        Number of forward passes completed.
    memory_stream : torch.cuda.Stream
        CUDA stream for memory operations.
    compute_done : torch.cuda.Event
        CUDA event signaling compute completion.
    memory_done : torch.cuda.Event
        CUDA event signaling memory completion.
    """

    def __init__(
        self,
        blocks: list[nn.Module],
        device: str | torch.device = torch.device("cuda"),
        use_pin_memory: bool = True,
        on_gpu_modules: list[nn.Module] = [],
        num_blocks_on_gpu: int = 1,
        empty_cache_freq: int = 0,
    ):
        self.blocks = blocks
        self.use_pin_memory = use_pin_memory
        self.on_gpu_modules = on_gpu_modules
        self.num_blocks_on_gpu = num_blocks_on_gpu
        assert self.num_blocks_on_gpu > 0

        # Two streams: one for compute, one for memory operations, will be initialized in set_device
        self.memory_stream = None

        self.compute_done = torch.cuda.Event(blocking=False)
        self.memory_done = torch.cuda.Event(blocking=False)

        self.buffer_blocks = [copy.deepcopy(blocks[0]), copy.deepcopy(blocks[0])]
        self._copy_plan_cache: dict[int, PackedCopyPlan] = {}
        self._packed_block_parameter_slabs: list[PackedParameterSlabs | None] = [None] * len(self.blocks)
        self._packed_buffer_parameter_slabs: list[PackedParameterSlabs | None] = [None] * len(self.buffer_blocks)

        self.device = None
        self.set_device(device)

        self.current_block_idx = 0
        self.forward_counter = 0
        self.empty_cache_freq = empty_cache_freq

    def set_device(self, device: torch.device | str, force: bool = False):
        """
        Set the CUDA device for offloading and memory operations.
        It will move buffer blocks and on-GPU modules to the specified device and offload other blocks to CPU, optionally using pinned memory.

        Parameters
        ----------
        device : torch.device or str
            Target CUDA device.
        force : bool, optional
            If True, force re-initialization even if device is unchanged. Default is False.

        Raises
        ------
        AssertionError
            If the device is not a CUDA device.
        """
        if isinstance(device, str):
            device = torch.device(device)
        assert device.type == "cuda"
        if self.device == device and not force:
            return
        self.device = device
        self._copy_plan_cache.clear()
        self.memory_stream = torch.cuda.Stream(device=device)
        for block in self.buffer_blocks:
            block.to(device)
        for module in self.on_gpu_modules:
            module.to(device)
        self._packed_buffer_parameter_slabs = [
            self._pack_module_parameters(block, device=device, pin_memory=False) for block in self.buffer_blocks
        ]
        for i, block in enumerate(self.blocks):
            if i < self.num_blocks_on_gpu:
                block.to(device)
                self._packed_block_parameter_slabs[i] = None
            else:
                block.to("cpu")
                self._packed_block_parameter_slabs[i] = self._pack_module_parameters(
                    block, device=torch.device("cpu"), pin_memory=self.use_pin_memory
                )
                if self.use_pin_memory:
                    for b in block.buffers(recurse=True):
                        b.data = b.data.pin_memory()

    @staticmethod
    def _should_skip_parameter_copy(name: str) -> bool:
        # Qwen's SVDQ runtime only consumes `smooth_factor`; the `_orig` copy is
        # kept for checkpoint compatibility and does not need to be reloaded.
        return name.endswith("smooth_factor_orig")

    def _iter_copyable_named_parameters(self, module: nn.Module):
        for name, parameter in module.named_parameters():
            if self._should_skip_parameter_copy(name):
                continue
            yield name, parameter

    def _pack_module_parameters(
        self,
        module: nn.Module,
        *,
        device: torch.device,
        pin_memory: bool,
    ) -> PackedParameterSlabs:
        grouped_params: dict[torch.dtype, list[tuple[str, nn.Parameter]]] = {}
        for name, parameter in self._iter_copyable_named_parameters(module):
            grouped_params.setdefault(parameter.dtype, []).append((name, parameter))

        packed_slabs: PackedParameterSlabs = {}
        with torch.no_grad():
            for dtype, items in grouped_params.items():
                total_numel = sum(int(parameter.numel()) for _, parameter in items)
                if total_numel == 0:
                    continue
                slab = torch.empty(total_numel, dtype=dtype, device=device)
                if device.type == "cpu" and pin_memory:
                    slab = slab.pin_memory()
                offset = 0
                for _, parameter in items:
                    next_offset = offset + int(parameter.numel())
                    view = slab[offset:next_offset].view_as(parameter)
                    view.copy_(parameter, non_blocking=False)
                    parameter.data = view
                    offset = next_offset
                packed_slabs[dtype] = slab
        return packed_slabs

    def _get_copy_plan(self, block_idx: int) -> PackedCopyPlan:
        cached_plan = self._copy_plan_cache.get(block_idx)
        if cached_plan is not None:
            return cached_plan

        buffer_idx = block_idx % len(self.buffer_blocks)
        src = self.blocks[block_idx]
        dst = self.buffer_blocks[buffer_idx]
        src_slabs = self._packed_block_parameter_slabs[block_idx]
        dst_slabs = self._packed_buffer_parameter_slabs[buffer_idx]
        assert src_slabs is not None
        assert dst_slabs is not None
        slab_pairs: list[TensorPair] = []
        for dtype in sorted(src_slabs.keys(), key=str):
            assert dtype in dst_slabs
            slab_pairs.append((src_slabs[dtype], dst_slabs[dtype]))

        buffer_pairs: list[TensorPair] = []
        for (src_name, src_buffer), (dst_name, dst_buffer) in zip(src.named_buffers(), dst.named_buffers()):
            assert src_name == dst_name
            buffer_pairs.append((src_buffer, dst_buffer))

        module_pairs: list[ModulePair] = []
        for src_module, dst_module in zip(src.modules(), dst.modules()):
            if hasattr(src_module, "wtscale"):
                assert hasattr(dst_module, "wtscale")
                module_pairs.append((src_module, dst_module))
            else:
                assert not hasattr(dst_module, "wtscale")

        cached_plan = (tuple(slab_pairs), tuple(buffer_pairs), tuple(module_pairs))
        self._copy_plan_cache[block_idx] = cached_plan
        return cached_plan

    def load_block(self, block_idx: int, non_blocking: bool = True):
        """
        Move a transformer block from CPU to GPU buffer.

        Parameters
        ----------
        block_idx : int
            Index of the block to load.
        non_blocking : bool, optional
            Whether to use non-blocking memory copy. Default is True.

        Notes
        -----
        - No action is taken if the block is already on GPU or index is out of range.
        """
        # if the block is already on GPU, don't load it to the buffer
        if block_idx < self.num_blocks_on_gpu:
            return
        # if there are blocks on GPU, don't load the first block to the buffer again
        if block_idx >= len(self.blocks):
            return

        buffer_idx = block_idx % len(self.buffer_blocks)
        copy_plan = self._get_copy_plan(block_idx)
        copy_params_into(
            self.blocks[block_idx],
            self.buffer_blocks[buffer_idx],
            non_blocking=non_blocking,
            copy_plan=copy_plan,
        )

    def step(self, compute_stream: torch.cuda.Stream | None = None):
        """
        Advance to the next transformer block, triggering asynchronous preloading.

        It will preload the next block onto GPU in the background and synchronize between compute and memory streams.
        After all the blocks are processed, it will call torch.cuda.empty_cache() periodically if ``empty_cache_freq`` > 0.

        Parameters
        ----------
        compute_stream : torch.cuda.Stream, optional
            CUDA stream for compute operations. If None, uses current stream.
        """
        if compute_stream is None:
            compute_stream = torch.cuda.current_stream()
        next_compute_done = torch.cuda.Event()
        next_compute_done.record(compute_stream)
        with torch.cuda.stream(self.memory_stream):
            self.memory_stream.wait_event(self.compute_done)
            self.load_block(self.current_block_idx + 1)  # if the current block is the last block, load the first block
            next_memory_done = torch.cuda.Event()
            next_memory_done.record(self.memory_stream)
        self.memory_done = next_memory_done
        self.compute_done = next_compute_done
        self.current_block_idx += 1
        if self.current_block_idx < len(self.blocks):
            # get ready for the next compute
            compute_stream.wait_event(self.memory_done)
        else:
            # ready to finish
            compute_stream.wait_event(self.compute_done)
            self.current_block_idx = 0
            self.forward_counter += 1
            if self.empty_cache_freq > 0 and self.forward_counter % self.empty_cache_freq == 0:
                torch.cuda.empty_cache()

    def get_block(self, block_idx: int | None = None) -> nn.Module:
        """
        Retrieve the current or specified transformer block for computation.
        It will return a buffer block if the requested block is offloaded.

        Parameters
        ----------
        block_idx : int, optional
            Index of the block to retrieve. If None, returns the current block.

        Returns
        -------
        block : nn.Module
            The requested transformer block (on GPU if needed).
        """
        if block_idx is None:
            block_idx = self.current_block_idx
        if block_idx < self.num_blocks_on_gpu:
            return self.blocks[block_idx]
        else:
            return self.buffer_blocks[block_idx % len(self.buffer_blocks)]

    def initialize(self, stream: torch.cuda.Stream | None = None):
        """
        Initialize CUDA events for compute and memory streams.
        It will record the initial events for the compute and memory streams.

        Parameters
        ----------
        stream : torch.cuda.Stream, optional
            CUDA stream to record initial events. If None, uses current stream.

        Notes
        -----
        - Should be called before the first forward pass.
        """
        if stream is None:
            stream = torch.cuda.current_stream()
        self.compute_done.record(stream)
        self.memory_done.record(stream)

