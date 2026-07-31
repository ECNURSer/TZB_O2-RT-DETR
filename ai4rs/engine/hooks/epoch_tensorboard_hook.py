from __future__ import annotations

from collections import defaultdict

import torch
from mmengine.dist import all_reduce, get_dist_info
from mmengine.hooks import Hook

from ai4rs.registry import HOOKS


@HOOKS.register_module()
class EpochTensorboardScalarHook(Hook):
    """Log train and validation losses as epoch-level TensorBoard scalars."""

    priority = "LOW"

    def __init__(
        self,
        log_val_loss: bool = True,
        loss_keys: tuple[str, ...] = ("loss_bbox", "loss_cls", "loss_iou"),
    ) -> None:
        self.log_val_loss = log_val_loss
        self.loss_keys = set(loss_keys)
        self._train_sums: dict[str, float] = defaultdict(float)
        self._train_counts: dict[str, int] = defaultdict(int)

    def before_train_epoch(self, runner) -> None:
        self._train_sums.clear()
        self._train_counts.clear()

    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs: dict | None = None) -> None:
        if outputs:
            self._accumulate(self._train_sums, self._train_counts, outputs)

    def after_train_epoch(self, runner) -> None:
        epoch = runner.epoch + 1
        train_scalars = self._sync_averages(runner, self._train_sums, self._train_counts, "train/")
        self._log_scalars(runner, train_scalars, epoch)
        if self.log_val_loss:
            val_scalars = self._compute_val_loss(runner)
            self._log_scalars(runner, val_scalars, epoch)

    def _compute_val_loss(self, runner) -> dict[str, float]:
        model = runner.model
        module = getattr(model, "module", model)
        original_training_states = {submodule: submodule.training for submodule in module.modules()}
        module.train()
        for submodule in module.modules():
            if isinstance(submodule, torch.nn.modules.batchnorm._BatchNorm):
                submodule.eval()
        sums: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        try:
            with torch.no_grad():
                for data_batch in runner.val_dataloader:
                    data = module.data_preprocessor(data_batch, training=True)
                    losses = module._run_forward(data, mode="loss")
                    _, log_vars = module.parse_losses(losses)
                    self._accumulate(sums, counts, log_vars)
        finally:
            for submodule, training in original_training_states.items():
                submodule.training = training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return self._sync_averages(runner, sums, counts, "val/")

    def _accumulate(self, sums: dict[str, float], counts: dict[str, int], values: dict) -> None:
        for key, value in values.items():
            if key not in self.loss_keys:
                continue
            scalar = self._to_float(value)
            if scalar is None:
                continue
            sums[key] += scalar
            counts[key] += 1

    def _sync_averages(
        self,
        runner,
        sums: dict[str, float],
        counts: dict[str, int],
        prefix: str,
    ) -> dict[str, float]:
        keys = sorted(key for key, count in counts.items() if count > 0)
        if not keys:
            return {}
        device = self._model_device(runner)
        sum_tensor = torch.tensor([sums[key] for key in keys], dtype=torch.float64, device=device)
        count_tensor = torch.tensor([counts[key] for key in keys], dtype=torch.float64, device=device)
        _, world_size = get_dist_info()
        if world_size > 1:
            all_reduce(sum_tensor, op="sum")
            all_reduce(count_tensor, op="sum")
        rank, _ = get_dist_info()
        if rank != 0:
            return {}
        return {
            f"{prefix}{key}": float((sum_tensor[index] / count_tensor[index]).item())
            for index, key in enumerate(keys)
            if count_tensor[index].item() > 0
        }

    def _log_scalars(self, runner, scalars: dict[str, float], epoch: int) -> None:
        rank, _ = get_dist_info()
        if rank != 0 or not scalars:
            return
        runner.visualizer.add_scalars(scalars, step=epoch)
        for key, value in scalars.items():
            runner.message_hub.update_scalar(key, value)

    def _model_device(self, runner) -> torch.device:
        model = getattr(runner.model, "module", runner.model)
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")

    @staticmethod
    def _to_float(value) -> float | None:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            return float(value.detach().mean().cpu())
        if isinstance(value, (int, float)):
            return float(value)
        return None
