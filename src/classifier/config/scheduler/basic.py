from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

from src.classifier.nn.schedule import MultiStepBS, Schedule

if TYPE_CHECKING:
    from torch.optim.lr_scheduler import ReduceLROnPlateau


@dataclass
class FixedStep(Schedule):
    """
    Use a much larger training batches and learning rate by default [1]_.

    .. [1] https://arxiv.org/pdf/1711.00489.pdf
    """

    epoch: int = 20

    bs_init: int = 2**10
    bs_scale: float = 2.0
    bs_milestones: list[int] = (1, 3, 6, 10)
    bs_kwargs: dict[str, Any] = None
    lr_init: float = 1.0e-2
    lr_scale: float = 0.25
    lr_milestones: list[int] = (15, 16, 17, 18, 19, 20, 21, 22, 23, 24)
    lr_kwargs: dict[str, Any] = None
    weight_decay: float = 0.0

    def __post_init__(self):
        self.bs_kwargs = self.bs_kwargs or {}
        self.lr_kwargs = self.lr_kwargs or {}
    
    def optimizer(self, parameters, **kwargs):
        import torch.optim as optim

        return optim.Adam(
            parameters,
            lr=self.lr_init,
            weight_decay=self.weight_decay,
            # amsgrad=True, # PLAN test performance
            **kwargs,
        )

    def bs_scheduler(self, dataset, **kwargs):
        return MultiStepBS(
            dataset=dataset,
            batch_size=self.bs_init,
            milestones=self.bs_milestones,
            gamma=self.bs_scale,
            **self.bs_kwargs | kwargs,
        )

    def lr_scheduler(self, optimizer, **kwargs):
        from torch.optim.lr_scheduler import MultiStepLR

        return MultiStepLR(
            optimizer=optimizer,
            milestones=self.lr_milestones,
            gamma=self.lr_scale,
            **self.lr_kwargs | kwargs,
        )


@dataclass
class AutoStep(FixedStep):
    require_benchmark = True

    lr_threshold: float = 1e-4
    lr_patience: int = 1
    lr_cooldown: int = 1
    lr_min: float = 2e-4
    lr_metric: Iterable[str] = ("benchmarks", "validation", "loss")

    def lr_scheduler(self, optimizer, **kwargs):
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        return ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=self.lr_scale,
            threshold=self.lr_threshold,
            patience=self.lr_patience,
            cooldown=self.lr_cooldown,
            min_lr=self.lr_min,
            **kwargs,
        )

    def lr_step(self, lr: ReduceLROnPlateau, benchmark: dict = None):
        # required=True here raises KeyError and kills the run when the benchmark
        # dict is keyed by validation-set name rather than the literal
        # "validation" (see EarlyStopStep._validation_loss). Skip the LR step
        # instead: a missed plateau update must not abort training.
        metric = self._get_key(self.lr_metric, benchmark, value_type=float)
        if metric is None:
            import logging

            logging.warning(
                f"{type(self).__name__}: validation loss unavailable this epoch; "
                "skipping ReduceLROnPlateau step"
            )
            return
        lr.step(metric, self._get_key(self.epoch_key, benchmark))


@dataclass
class EarlyStopStep(AutoStep):
    """:class:`AutoStep` that also stops once the validation loss stops improving.

    ``epoch`` becomes an upper bound rather than a fixed cost: training ends after
    ``es_patience`` consecutive epochs without an improvement larger than
    ``es_min_delta``. Requires benchmarks (i.e. ``Monitor`` enabled), inherited
    from :class:`AutoStep` via ``require_benchmark``.
    """

    es_patience: int = 3
    es_min_delta: float = 1e-4
    es_min_epoch: int = 5

    _es_best: float = None
    _es_wait: int = 0
    _es_stopped: int = None

    def _validation_loss(self, benchmark: dict) -> float | None:
        """Validation loss, tolerating how the benchmark dict is actually keyed.

        ``_iter_benchmark`` keys results by validation-set *name*
        (``benchmark["benchmarks"][<set>][<scalar>]``), so the ``AutoStep``
        default path ``("benchmarks", "validation", "loss")`` only resolves when a
        set happens to be called "validation". Try that first, then fall back to
        the loss of any set that reports one.
        """
        loss = self._get_key(self.lr_metric, benchmark, value_type=float)
        if loss is not None:
            return loss
        sets = self._get_key(("benchmarks",), benchmark)
        if not isinstance(sets, dict):
            return None
        losses = []
        for v in sets.values():
            if not isinstance(v, dict):
                continue
            l = v.get("scalars", {}).get("loss") if "scalars" in v and isinstance(v["scalars"], dict) else v.get("loss")
            if isinstance(l, (int, float)):
                losses.append(l)
        return float(sum(losses) / len(losses)) if losses else None

    def lr_step(self, lr: ReduceLROnPlateau, benchmark: dict = None):
        metric = self._validation_loss(benchmark)
        if metric is None:
            return
        lr.step(metric, self._get_key(self.epoch_key, benchmark))

    def should_stop(self, benchmark: dict = None) -> bool:
        import logging

        # Never let a metric lookup abort training: early stopping is an
        # optimisation, so on any unexpected benchmark shape just keep going.
        try:
            epoch = self._get_key(self.epoch_key, benchmark)
            loss = self._validation_loss(benchmark)
        except Exception:
            logging.warning(
                "EarlyStopStep: could not read validation loss from benchmark; "
                "continuing without early stopping",
                exc_info=True,
            )
            return False
        if loss is None or epoch is None:
            return False
        if self._es_best is None or loss < self._es_best - self.es_min_delta:
            self._es_best = loss
            self._es_wait = 0
            return False
        self._es_wait += 1
        if epoch < self.es_min_epoch or self._es_wait < self.es_patience:
            return False
        self._es_stopped = epoch
        logging.info(
            f"Early stopping at epoch {epoch}/{self.epoch}: validation loss did not "
            f"improve by >{self.es_min_delta:g} for {self._es_wait} epochs "
            f"(best={self._es_best:.6g})"
        )
        return True
