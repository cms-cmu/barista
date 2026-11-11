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

    # Weight averaging (src/classifier/ml/training.py:_WeightAverage): "swa" (uniform mean of the
    # end-of-epoch weights) or "ema" (per-step exponential moving average) over epochs
    # [average_start, average_end]; after epoch average_end (default: the last) the average replaces
    # the live weights and training, if any epochs remain, continues from it.
    average: str = None
    average_start: int = 1
    average_end: int = None
    ema_decay: float = 0.999

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
class ClosureGatedStep(FixedStep):
    """:class:`FixedStep` whose learning-rate decay starts when the model's calibration is good.

    The learning rate stays at ``lr_init`` until the sideband closure of the fit-quality benchmark
    (FvTFitQuality: 4b data vs source x w + 4b ttbar, histogram ``gate_hist``) on the TRAINING split
    has an excess (chi2/ndf - 1) per 100k target events of at most ``gate_max_excess``; then it
    decays by ``lr_scale`` per epoch for ``n_decay`` epochs and the stage ends. The training split,
    not the validation one: in a k-fold FvT the validation fold is the events the model is applied
    to, and tuning on their closure would bias it. The gate is evaluated from ``gate_min_epoch`` on;
    at ``gate_max_epoch`` the decay starts regardless (logged). ``epoch`` is set to
    gate_max_epoch + n_decay, the longest possible stage. Needs the fit-quality benchmark (a model
    with fit_quality, e.g. HCR FvT/MvD) and benchmarks enabled.
    """

    require_benchmark = True

    gate_hist: str = "fvt_sb"
    gate_split: str = "training"
    gate_max_excess: float = 1.0
    gate_min_epoch: int = 8
    gate_max_epoch: int = 30
    n_decay: int = 5

    _gate: int = None

    def __post_init__(self):
        super().__post_init__()
        self.epoch = int(self.gate_max_epoch) + int(self.n_decay)

    def _factor(self, epoch: int) -> float:
        if self._gate is None or epoch < self._gate:
            return 1.0
        return self.lr_scale ** (epoch - self._gate + 1)

    def lr_scheduler(self, optimizer, **kwargs):
        from torch.optim.lr_scheduler import LambdaLR

        # called once at the start of each stage: a schedule object reused for another fold's
        # training must not inherit its gate (a resume restores it afterwards from the checkpoint)
        self._gate = None
        return LambdaLR(optimizer=optimizer, lr_lambda=self._factor, **self.lr_kwargs | kwargs)

    def _excess(self, benchmark: dict) -> float | None:
        fqs = self._get_key(("benchmarks", self.gate_split, "fit_quality"), benchmark)
        try:
            h = fqs[0]["histograms"][self.gate_hist]
            n = float(sum(h["target"]))
            if h.get("chi2_ndf") is None or n <= 0:
                return None
            return (float(h["chi2_ndf"]) - 1.0) / (n / 1e5)
        except Exception:
            return None

    def lr_step(self, lr, benchmark: dict = None):
        import logging

        epoch = self._get_key(self.epoch_key, benchmark)
        if self._gate is None and epoch is not None:
            excess = self._excess(benchmark)
            shown = "n/a" if excess is None else f"{excess:.3f}"
            if epoch >= self.gate_min_epoch and excess is not None and excess <= self.gate_max_excess:
                self._gate = epoch
                logging.info(
                    f"ClosureGatedStep: epoch {epoch}: {self.gate_split} {self.gate_hist} excess "
                    f"{shown}/100k <= {self.gate_max_excess}: learning-rate decay starts "
                    f"({self.n_decay} epochs)"
                )
            elif epoch >= self.gate_max_epoch:
                self._gate = epoch
                logging.warning(
                    f"ClosureGatedStep: epoch {epoch}: closure gate never passed (last excess "
                    f"{shown}/100k > {self.gate_max_excess}); learning-rate decay starts anyway"
                )
            else:
                logging.info(
                    f"ClosureGatedStep: epoch {epoch}: {self.gate_split} {self.gate_hist} excess "
                    f"{shown}/100k (gate {self.gate_max_excess}, from epoch {self.gate_min_epoch})"
                )
        lr.step(epoch)

    def should_stop(self, benchmark: dict = None) -> bool:
        epoch = self._get_key(self.epoch_key, benchmark)
        return self._gate is not None and epoch is not None and epoch >= self._gate + self.n_decay

    def state_dict(self) -> dict:
        return {"gate": self._gate}

    def load_state_dict(self, state: dict):
        self._gate = (state or {}).get("gate")


@dataclass
class CosineAnnealing(FixedStep):
    """
    Cosine annealing learning rate schedule with multi-step batch size scaling.
    """

    eta_min: float = 1e-6

    def lr_scheduler(self, optimizer, **kwargs):
        from torch.optim.lr_scheduler import CosineAnnealingLR

        return CosineAnnealingLR(
            optimizer=optimizer,
            T_max=self.epoch,
            eta_min=self.eta_min,
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
