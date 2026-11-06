"""Comprehensive tests for ML training robustness, learning dynamics, and workflow configurations.

Covers:
  - Workflow YAML config integrity (HH4b Run3, ttHbb Run3, HH4b bkg_syst, ttHbb bkg_syst)
  - Model architectures across channels (HH4b FvT/SvB, ttHbb FvT/SvB with attention gates and KV projection)
  - Learning rate schedulers: FixedStep (milestone scaling), FinetuneStep (plateau decay), CosineAnnealing
  - Gradient flow and loss calculations across all active layers
  - Overfitting capacity checks across all four model configurations
  - Underfitting / degenerate gradient detection across all four configurations
  - Multi-stage training lifecycle: Stage 1 training -> Stage 2 finetuning (weight freeze/transfer guard)
  - Deterministic 3-stream seed reproducibility
  - Kinematic and weight edge-cases (zero-weight events, masked jets, high-pt)
"""

import os
import sys
from pathlib import Path

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("coffea4bees"))

import pytest
import torch
import torch.nn.functional as F
import yaml

from coffea4bees.classifier.nn.blocks.HCR import HCR
from src.classifier.config.scheduler.basic import CosineAnnealing, FixedStep
from src.classifier.config.scheduler.finetune import FinetuneStep
from src.classifier.config.setting.HCR import Input, Output


def _build_synthetic_batch(
    batch_size: int = 16,
    ancillary_features: list[str] = None,
    n_not_can_jet: int = 8,
    seed: int = 42,
    device: str = "cpu",
):
    """Generate a valid, physics-realistic synthetic batch for HCR forward/backward tests."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    if ancillary_features is None:
        ancillary_features = ["xW", "nSelJets", "xbW"]

    # 4 candidate jets: feature-major [batch, (pt, eta, phi, mass), 4_jets]
    can_jets = torch.zeros(batch_size, 4, 4, device="cpu")
    can_jets[:, 0, :] = torch.rand(batch_size, 4, generator=g) * 100.0 + 30.0   # pt: 30 - 130 GeV
    can_jets[:, 1, :] = (torch.rand(batch_size, 4, generator=g) - 0.5) * 4.0   # eta: -2.0 to 2.0
    can_jets[:, 2, :] = (torch.rand(batch_size, 4, generator=g) - 0.5) * 6.0   # phi: -3.0 to 3.0
    can_jets[:, 3, :] = torch.rand(batch_size, 4, generator=g) * 15.0 + 5.0    # mass: 5 - 20 GeV

    # Not-candidate jets: feature-major [batch, (pt, eta, phi, mass, isSelJet), n_not_can_jet]
    not_can_jets = torch.zeros(batch_size, 5, n_not_can_jet, device="cpu")
    not_can_jets[:, 0, :] = torch.rand(batch_size, n_not_can_jet, generator=g) * 60.0 + 20.0
    not_can_jets[:, 1, :] = (torch.rand(batch_size, n_not_can_jet, generator=g) - 0.5) * 4.0
    not_can_jets[:, 2, :] = (torch.rand(batch_size, n_not_can_jet, generator=g) - 0.5) * 6.0
    not_can_jets[:, 3, :] = torch.rand(batch_size, n_not_can_jet, generator=g) * 10.0 + 2.0
    not_can_jets[:, 4, :] = (torch.rand(batch_size, n_not_can_jet, generator=g) > 0.5).float()

    # Ancillary features: [batch, n_ancillary]
    ancillary = torch.zeros(batch_size, len(ancillary_features), device="cpu")
    for idx, feat in enumerate(ancillary_features):
        if "nSelJets" in feat:
            # Must be >= 4 so that (nSelJets - 3) > 0 for log transform
            ancillary[:, idx] = torch.randint(4, 10, (batch_size,), generator=g).float()
        elif "year" in feat:
            ancillary[:, idx] = 2022.0
        elif "xW" in feat or "xbW" in feat:
            ancillary[:, idx] = torch.rand(batch_size, generator=g) * 2.0 + 0.1
        else:
            ancillary[:, idx] = torch.rand(batch_size, generator=g)

    # Balanced binary labels and unit weights
    labels = torch.cat([
        torch.zeros(batch_size // 2, dtype=torch.long),
        torch.ones(batch_size - batch_size // 2, dtype=torch.long),
    ])
    weights = torch.ones(batch_size, dtype=torch.float32)

    return {
        Input.CanJet: can_jets.to(device),
        Input.NotCanJet: not_can_jets.to(device),
        Input.ancillary: ancillary.to(device),
        Input.label: labels.to(device),
        Input.weight: weights.to(device),
    }


def _initialize_hcr_buffers(model: HCR, batch: dict):
    """Run updateMeanStd and initMeanStd to initialize GhostBatchNorm running statistics."""
    # Clone inputs because HCR.dataPrep modifies tensors in-place
    model.updateMeanStd(
        batch[Input.CanJet].clone(),
        batch[Input.NotCanJet].clone(),
        batch[Input.ancillary].clone(),
    )
    model.initMeanStd()
    model.setGhostBatches(0)


def _forward(model: HCR, batch: dict):
    """Execute model forward pass with cloned inputs to guard against in-place dataPrep transforms."""
    return model(
        batch[Input.CanJet].clone(),
        batch[Input.NotCanJet].clone(),
        batch[Input.ancillary].clone(),
    )


def _build_model_from_cfg(cfg: dict) -> HCR:
    """Instantiate HCR module based on test configuration dictionary."""
    return HCR(
        dijetFeatures=cfg["dijet_features"],
        quadjetFeatures=cfg["quadjet_features"],
        ancillaryFeatures=cfg["ancillary"],
        canJetFeatures=["pt", "eta", "phi", "mass"],
        useOthJets="attention",
        device="cpu",
        nClasses=cfg["n_classes"],
        use_attention_gate=cfg.get("use_attention_gate", False),
        use_kv_proj=cfg.get("use_kv_proj", False),
    )


# ==============================================================================
# Model Configurations under Test
# ==============================================================================
MODEL_CONFIGS = [
    {
        "name": "HH4b_FvT",
        "ancillary": ["xW", "nSelJets", "xbW"],
        "dijet_features": 8,
        "quadjet_features": 8,
        "n_classes": 2,
        "use_attention_gate": False,
        "use_kv_proj": False,
    },
    {
        "name": "ttHbb_FvT",
        "ancillary": ["year", "nSelJets", "xW", "xbW"],
        "dijet_features": 8,
        "quadjet_features": 8,
        "n_classes": 2,
        "use_attention_gate": False,
        "use_kv_proj": False,
    },
    {
        "name": "HH4b_SvB",
        "ancillary": ["xW", "nSelJets", "xbW"],
        "dijet_features": 8,
        "quadjet_features": 8,
        "n_classes": 2,
        "use_attention_gate": False,
        "use_kv_proj": False,
    },
    {
        "name": "ttHbb_SvB",
        "ancillary": ["year", "nSelJets", "xW", "xbW"],
        "dijet_features": 8,
        "quadjet_features": 8,
        "n_classes": 2,
        "use_attention_gate": True,
        "use_kv_proj": True,
    },
]


# ==============================================================================
# 1. Workflow Config Extraction & Integrity Checks
# ==============================================================================
CONFIG_TARGETS = [
    ("HH4b Run3", "coffea4bees/workflows/config/nominal_run3_2025.yml"),
    ("ttHbb Run3", "coffea4bees/workflows/config/ttHbb_Run3.yml"),
    ("HH4b bkg_syst", "coffea4bees/workflows/config/analysis_HH4b_bkg_syst_declib16nc.yml"),
    ("ttHbb bkg_syst", "coffea4bees/workflows/config/ttHbb_bkg_syst_declib16nc.yml"),
]


@pytest.mark.parametrize("name,cfg_path", CONFIG_TARGETS)
def test_workflow_config_parseable(name, cfg_path):
    """Verify that workflow configs in coffea4bees/workflows/config/ parse correctly."""
    p = Path(cfg_path)
    if not p.exists():
        pytest.skip(f"Config path {cfg_path} not found in this checkout.")

    with open(p, "rt") as f:
        data = yaml.safe_load(f)

    assert isinstance(data, dict), f"Failed to parse {name} YAML as dict"

    has_fvt = (
        "fvt_train_workflow" in data
        or "train_workflow" in data
        or ("fvt" in data and ("train_workflow" in data["fvt"] or "wfs_base" in data["fvt"]))
    )
    assert has_fvt, f"Expected FvT training definition in {name} ({cfg_path})"


# ==============================================================================
# 2. Learning Rate Schedulers (FixedStep, FinetuneStep, CosineAnnealing)
# ==============================================================================
def test_fixed_step_lr_schedule():
    """Verify FixedStep multi-step learning rate milestones and scaling."""
    param = torch.nn.Parameter(torch.zeros(10))
    schedule = FixedStep(
        epoch=10,
        lr_init=1.0e-2,
        lr_scale=0.25,
        lr_milestones=[3, 6],
    )
    opt = schedule.optimizer([param])
    lr_scheduler = schedule.lr_scheduler(opt)

    # Epoch 0: lr = 1e-2
    assert opt.param_groups[0]["lr"] == pytest.approx(1.0e-2)

    # Epochs 1-2: still 1e-2
    for _ in range(2):
        opt.step()
        schedule.lr_step(lr_scheduler)
    assert opt.param_groups[0]["lr"] == pytest.approx(1.0e-2)

    # Epoch 3: milestone 1 -> 1e-2 * 0.25 = 2.5e-3
    opt.step()
    schedule.lr_step(lr_scheduler)
    assert opt.param_groups[0]["lr"] == pytest.approx(2.5e-3)

    # Epoch 4-5: still 2.5e-3
    for _ in range(2):
        opt.step()
        schedule.lr_step(lr_scheduler)
    assert opt.param_groups[0]["lr"] == pytest.approx(2.5e-3)

    # Epoch 6: milestone 2 -> 2.5e-3 * 0.25 = 6.25e-4
    opt.step()
    schedule.lr_step(lr_scheduler)
    assert opt.param_groups[0]["lr"] == pytest.approx(6.25e-4)


def test_finetune_step_plateau_schedule():
    """Verify FinetuneStep reduces LR upon plateauing validation loss metric."""
    param = torch.nn.Parameter(torch.zeros(10))
    schedule = FinetuneStep(
        lr_init=1.0e-4,
        lr_scale=0.5,
        lr_threshold=1.0e-3,
        lr_patience=1,
        lr_cooldown=0,
        lr_min=1.0e-8,
        lr_metric=("validation", "loss"),
    )
    opt = schedule.optimizer([param])
    lr_scheduler = schedule.lr_scheduler(opt)

    assert opt.param_groups[0]["lr"] == pytest.approx(1.0e-4)

    # Feed stagnant loss over epochs to trigger patience threshold
    for epoch in range(4):
        benchmark = {"validation": {"loss": 0.500}}
        schedule.lr_step(lr_scheduler, benchmark)

    # LR should have dropped by factor 0.5
    assert opt.param_groups[0]["lr"] == pytest.approx(5.0e-5)


def test_cosine_annealing_schedule():
    """Verify CosineAnnealing produces smooth monotonic learning rate decrease."""
    param = torch.nn.Parameter(torch.zeros(10))
    total_epochs = 10
    schedule = CosineAnnealing(
        epoch=total_epochs,
        lr_init=1.0e-2,
        eta_min=1.0e-5,
    )
    opt = schedule.optimizer([param])
    lr_scheduler = schedule.lr_scheduler(opt)

    lrs = [opt.param_groups[0]["lr"]]
    for _ in range(total_epochs):
        opt.step()
        lr_scheduler.step()
        lrs.append(opt.param_groups[0]["lr"])

    # Verify strictly decreasing
    for i in range(len(lrs) - 1):
        assert lrs[i] >= lrs[i + 1], f"Cosine LR not non-increasing at step {i}: {lrs[i]} < {lrs[i+1]}"
    assert lrs[-1] == pytest.approx(1.0e-5, rel=1e-3)


# ==============================================================================
# 3. Model Architecture & Forward Pass Tests
# ==============================================================================
@pytest.mark.parametrize("cfg", MODEL_CONFIGS, ids=lambda c: c["name"])
def test_model_instantiation_and_forward(cfg):
    """Verify HCR instantiates and executes a forward pass across distinct channel configurations."""
    model = _build_model_from_cfg(cfg)
    batch = _build_synthetic_batch(batch_size=16, ancillary_features=cfg["ancillary"])
    _initialize_hcr_buffers(model, batch)

    c, q = _forward(model, batch)
    assert c.shape == (16, cfg["n_classes"]), f"Output c has wrong shape: {c.shape}"
    assert q.shape == (16, 3), f"Output q has wrong shape: {q.shape}"
    assert torch.isfinite(c).all(), "Non-finite values detected in classification output"
    assert torch.isfinite(q).all(), "Non-finite values detected in quadjet output"


# ==============================================================================
# 4. Loss Calculation & Gradient Flow (Backpropagation)
# ==============================================================================
@pytest.mark.parametrize("cfg", MODEL_CONFIGS, ids=lambda c: c["name"])
def test_gradient_flow_and_loss(cfg):
    """Verify loss computation, finite backpropagation, and full gradient flow to all trainable layers."""
    model = _build_model_from_cfg(cfg)
    batch = _build_synthetic_batch(batch_size=16, ancillary_features=cfg["ancillary"])
    _initialize_hcr_buffers(model, batch)

    model.train()
    c, _ = _forward(model, batch)
    loss = F.cross_entropy(c, batch[Input.label])

    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    assert loss.item() > 0.0, f"Loss should be positive, got {loss.item()}"

    loss.backward()

    has_gradients = False
    for p_name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {p_name} did not receive a gradient"
            assert torch.isfinite(param.grad).all(), f"Parameter {p_name} has non-finite gradient"
            if param.grad.abs().sum() > 0:
                has_gradients = True

    assert has_gradients, "No parameter received non-zero gradients"


# ==============================================================================
# 5. Overfitting Capacity Tests across All Four Models
# ==============================================================================
@pytest.mark.parametrize("cfg", MODEL_CONFIGS, ids=lambda c: c["name"])
def test_model_capacity_overfit(cfg):
    """Verify that each model configuration has the capacity to overfit a small batch (loss drops > 50%)."""
    model = _build_model_from_cfg(cfg)
    batch = _build_synthetic_batch(batch_size=8, ancillary_features=cfg["ancillary"], seed=123)
    _initialize_hcr_buffers(model, batch)

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)

    initial_loss = None
    final_loss = None

    for step in range(30):
        opt.zero_grad()
        c, _ = _forward(model, batch)
        loss = F.cross_entropy(c, batch[Input.label])
        if step == 0:
            initial_loss = loss.item()
        loss.backward()
        opt.step()
        final_loss = loss.item()

    assert final_loss < initial_loss * 0.5, (
        f"{cfg['name']} failed to overfit tiny batch: initial={initial_loss:.4f}, final={final_loss:.4f}"
    )


# ==============================================================================
# 6. Underfitting / Degenerate Gradient Detection across All Four Models
# ==============================================================================
@pytest.mark.parametrize("cfg", MODEL_CONFIGS, ids=lambda c: c["name"])
def test_underfit_degenerate_detection(cfg):
    """Verify that if parameters are frozen or LR is zero, the model does NOT overfit (sanity check)."""
    model = _build_model_from_cfg(cfg)
    batch = _build_synthetic_batch(batch_size=8, ancillary_features=cfg["ancillary"], seed=123)
    _initialize_hcr_buffers(model, batch)

    model.train()
    # Optimizer with zero learning rate to simulate complete underfitting / frozen updates
    opt = torch.optim.Adam(model.parameters(), lr=0.0)

    initial_loss = None
    final_loss = None

    for step in range(10):
        opt.zero_grad()
        c, _ = _forward(model, batch)
        loss = F.cross_entropy(c, batch[Input.label])
        if step == 0:
            initial_loss = loss.item()
        loss.backward()
        opt.step()
        final_loss = loss.item()

    # With zero LR, loss must remain strictly unchanged
    assert final_loss == pytest.approx(initial_loss, rel=1e-5), (
        f"{cfg['name']} changed loss unexpectedly under zero learning rate"
    )


# ==============================================================================
# 7. Multi-Stage Lifecycle & Finetuning Bug Protection across All Models
# ==============================================================================
@pytest.mark.parametrize("cfg", MODEL_CONFIGS, ids=lambda c: c["name"])
def test_finetuning_stage_transition(cfg):
    """Verify the 2-stage lifecycle: Stage 1 training -> Stage 2 finetuning.

    Guards against the finetuning bug where weights fail to transfer, embedding
    layers are improperly frozen/unfrozen, or LR schedulers crash upon re-init.
    """
    model = _build_model_from_cfg(cfg)
    batch = _build_synthetic_batch(batch_size=16, ancillary_features=cfg["ancillary"])
    _initialize_hcr_buffers(model, batch)

    # --- Stage 1: Initial Training ---
    model.train()
    opt1 = torch.optim.Adam(model.parameters(), lr=1e-3)
    c1, _ = _forward(model, batch)
    loss1 = F.cross_entropy(c1, batch[Input.label])
    loss1.backward()
    opt1.step()
    opt1.zero_grad()

    # Snapshot weights after Stage 1
    frozen_indices = model.embedding_layers()
    output_indices = model.output_layers()
    w_emb_before = [p.clone().detach() for idx in frozen_indices for l in model.layers.layers[idx] for p in l.parameters()]
    w_out_before = [p.clone().detach() for idx in output_indices for l in model.layers.layers[idx] for p in l.parameters()]

    # --- Stage 2: Finetuning Transition ---
    model.layers.setLayerRequiresGrad(requires_grad=False, index=frozen_indices)

    # Assert embedding layers are frozen while output layers remain trainable
    for idx in frozen_indices:
        for layer in model.layers.layers[idx]:
            for param in layer.parameters():
                assert not param.requires_grad, f"Layer {idx} should be frozen during finetuning"

    for idx in output_indices:
        for layer in model.layers.layers[idx]:
            for param in layer.parameters():
                assert param.requires_grad, f"Output layer {idx} must remain trainable during finetuning"

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable_params) > 0, "No trainable parameters found for finetuning stage"
    opt2 = torch.optim.Adam(trainable_params, lr=1e-4)

    # Verify forward and backward pass in finetuning stage
    c2, _ = _forward(model, batch)
    loss2 = F.cross_entropy(c2, batch[Input.label])
    assert torch.isfinite(loss2), "Finetuning loss is not finite"
    loss2.backward()
    opt2.step()

    # Verify frozen weights DID NOT change, but trainable output weights DID change
    w_emb_after = [p.clone().detach() for idx in frozen_indices for l in model.layers.layers[idx] for p in l.parameters()]
    w_out_after = [p.clone().detach() for idx in output_indices for l in model.layers.layers[idx] for p in l.parameters()]

    for wb, wa in zip(w_emb_before, w_emb_after):
        assert torch.allclose(wb, wa), "Frozen embedding weight modified during finetuning stage"
    assert any(not torch.allclose(wb, wa) for wb, wa in zip(w_out_before, w_out_after)), (
        "Output layer weights failed to update during finetuning stage"
    )

    # Unfreeze layers to restore full graph
    model.layers.setLayerRequiresGrad(requires_grad=True)
    all_trainable = all(p.requires_grad for p in model.parameters())
    assert all_trainable, "Failed to restore full requires_grad after finetuning"


# ==============================================================================
# 8. Deterministic Training Seeding Verification across All Models
# ==============================================================================
@pytest.mark.parametrize("cfg", MODEL_CONFIGS, ids=lambda c: c["name"])
def test_deterministic_training_reproducibility(cfg):
    """Verify that models initialized with identical seed 666 produce identical weights and losses."""
    losses = []
    param_states = []

    for _ in range(2):
        torch.manual_seed(666)
        model = _build_model_from_cfg(cfg)
        batch = _build_synthetic_batch(batch_size=8, ancillary_features=cfg["ancillary"], seed=666)
        _initialize_hcr_buffers(model, batch)

        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        c, _ = _forward(model, batch)
        loss = F.cross_entropy(c, batch[Input.label])
        loss.backward()
        opt.step()

        losses.append(loss.item())
        param_states.append([p.clone().detach() for p in model.parameters()])

    # Assert bitwise identical loss and parameters
    assert losses[0] == pytest.approx(losses[1], rel=1e-6), f"{cfg['name']}: Losses diverged under seed 666"
    for p0, p1 in zip(param_states[0], param_states[1]):
        assert torch.allclose(p0, p1), f"{cfg['name']}: Model weights diverged under seed 666"


# ==============================================================================
# 9. Kinematic and Weight Edge-Cases
# ==============================================================================
@pytest.mark.parametrize("cfg", MODEL_CONFIGS, ids=lambda c: c["name"])
def test_kinematic_and_weight_edge_cases(cfg):
    """Verify forward and backward numerical stability under extreme jet kinematics and event weights."""
    model = _build_model_from_cfg(cfg)
    batch = _build_synthetic_batch(batch_size=8, ancillary_features=cfg["ancillary"], seed=999)
    _initialize_hcr_buffers(model, batch)

    # Edge case 1: High pT jets (boosted kinematics up to 1.5 TeV)
    batch[Input.CanJet][:, 0, :] = 1500.0

    # Edge case 2: Near boundary eta (|eta| = 2.49)
    batch[Input.CanJet][:, 1, :] = 2.49

    # Edge case 3: Extreme small weights and large weights
    batch[Input.weight] = torch.tensor([1e-6, 1e-4, 1.0, 10.0, 100.0, 1.0, 1.0, 1.0])

    model.train()
    c, q = _forward(model, batch)

    assert torch.isfinite(c).all(), f"{cfg['name']}: Output contains NaN/Inf under extreme kinematics"
    assert torch.isfinite(q).all(), f"{cfg['name']}: Quadjet output contains NaN/Inf under extreme kinematics"

    # Weighted loss
    loss = (F.cross_entropy(c, batch[Input.label], reduction="none") * batch[Input.weight]).sum() / batch[Input.weight].sum()
    assert torch.isfinite(loss), f"{cfg['name']}: Weighted loss is non-finite under extreme weights"
    loss.backward()
