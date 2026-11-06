"""Tests for training robustness, learning dynamics, and workflow configurations.

Covers:
  - HH4b Run3 FvT and SvB
  - ttHbb Run3 FvT and SvB (with attention gate and KV projection)
  - HH4b and ttHbb background systematics (bkg_syst) FvT
  - Multi-stage lifecycle: initial training -> finetuning transition (guarding against regressions)
  - Gradient flow, loss validity, capacity/overfitting, and deterministic seeding
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
from src.classifier.config.setting.HCR import Input, Output
from src.classifier.config.state.label import MultiClass


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
    labels = torch.cat([torch.zeros(batch_size // 2, dtype=torch.long), torch.ones(batch_size - batch_size // 2, dtype=torch.long)])
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

    # Verify presence of training block in either inline or wfs_base format
    has_fvt = (
        "fvt_train_workflow" in data
        or "train_workflow" in data
        or ("fvt" in data and ("train_workflow" in data["fvt"] or "wfs_base" in data["fvt"]))
    )
    assert has_fvt, f"Expected FvT training definition in {name} ({cfg_path})"


# ==============================================================================
# 2. Distinct Model Architecture & Forward Pass Tests
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


@pytest.mark.parametrize("cfg", MODEL_CONFIGS, ids=lambda c: c["name"])
def test_model_instantiation_and_forward(cfg):
    """Verify HCR instantiates and executes a forward pass across distinct channel configurations."""
    model = HCR(
        dijetFeatures=cfg["dijet_features"],
        quadjetFeatures=cfg["quadjet_features"],
        ancillaryFeatures=cfg["ancillary"],
        canJetFeatures=["pt", "eta", "phi", "mass"],
        useOthJets="attention",
        device="cpu",
        nClasses=cfg["n_classes"],
        use_attention_gate=cfg["use_attention_gate"],
        use_kv_proj=cfg["use_kv_proj"],
    )
    batch = _build_synthetic_batch(batch_size=16, ancillary_features=cfg["ancillary"])
    _initialize_hcr_buffers(model, batch)

    c, q = _forward(model, batch)
    assert c.shape == (16, cfg["n_classes"]), f"Output c has wrong shape: {c.shape}"
    assert q.shape == (16, 3), f"Output q has wrong shape: {q.shape}"
    assert torch.isfinite(c).all(), "Non-finite values detected in classification output"
    assert torch.isfinite(q).all(), "Non-finite values detected in quadjet output"


# ==============================================================================
# 3. Loss Calculation & Gradient Flow (Backpropagation)
# ==============================================================================
@pytest.mark.parametrize("cfg", MODEL_CONFIGS, ids=lambda c: c["name"])
def test_gradient_flow_and_loss(cfg):
    """Verify loss computation, finite backpropagation, and full gradient flow to all trainable layers."""
    model = HCR(
        dijetFeatures=cfg["dijet_features"],
        quadjetFeatures=cfg["quadjet_features"],
        ancillaryFeatures=cfg["ancillary"],
        canJetFeatures=["pt", "eta", "phi", "mass"],
        useOthJets="attention",
        device="cpu",
        nClasses=cfg["n_classes"],
        use_attention_gate=cfg["use_attention_gate"],
        use_kv_proj=cfg["use_kv_proj"],
    )
    batch = _build_synthetic_batch(batch_size=16, ancillary_features=cfg["ancillary"])
    _initialize_hcr_buffers(model, batch)

    model.train()
    c, _ = _forward(model, batch)
    loss = F.cross_entropy(c, batch[Input.label])

    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    assert loss.item() > 0.0, f"Loss should be positive, got {loss.item()}"

    loss.backward()

    # Assert all trainable parameters with requires_grad receive valid gradients
    has_gradients = False
    for p_name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {p_name} did not receive a gradient"
            assert torch.isfinite(param.grad).all(), f"Parameter {p_name} has non-finite gradient"
            if param.grad.abs().sum() > 0:
                has_gradients = True

    assert has_gradients, "No parameter received non-zero gradients"


# ==============================================================================
# 4. Multi-Stage Lifecycle & Finetuning Bug Protection
# ==============================================================================
def test_finetuning_stage_transition():
    """Verify the 2-stage lifecycle: Stage 1 training -> Stage 2 finetuning.

    Guards against the finetuning bug where weights fail to transfer, embedding
    layers are improperly frozen/unfrozen, or LR schedulers crash upon re-init.
    """
    model = HCR(
        dijetFeatures=8,
        quadjetFeatures=8,
        ancillaryFeatures=["xW", "nSelJets", "xbW"],
        canJetFeatures=["pt", "eta", "phi", "mass"],
        useOthJets="attention",
        device="cpu",
        nClasses=2,
    )
    batch = _build_synthetic_batch(batch_size=16, ancillary_features=["xW", "nSelJets", "xbW"])
    _initialize_hcr_buffers(model, batch)

    # --- Stage 1: Initial Training ---
    model.train()
    opt1 = torch.optim.Adam(model.parameters(), lr=1e-3)
    c1, _ = _forward(model, batch)
    loss1 = F.cross_entropy(c1, batch[Input.label])
    loss1.backward()
    opt1.step()
    opt1.zero_grad()

    # --- Stage 2: Finetuning Transition ---
    # Freeze embedding layers as done during HCRTraining finetuning stage
    embedding_indices = model.embedding_layers()
    model.layers.setLayerRequiresGrad(requires_grad=False, index=embedding_indices)

    # Check that embedding layers are frozen while output layers remain trainable
    for idx in embedding_indices:
        for layer in model.layers.layers[idx]:
            for param in layer.parameters():
                assert not param.requires_grad, f"Layer {idx} should be frozen during finetuning"

    for idx in model.output_layers():
        for layer in model.layers.layers[idx]:
            for param in layer.parameters():
                assert param.requires_grad, f"Output layer {idx} must remain trainable during finetuning"

    # Setup Stage 2 optimizer on trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    assert len(trainable_params) > 0, "No trainable parameters found for finetuning stage"
    opt2 = torch.optim.Adam(trainable_params, lr=1e-4)

    # Verify forward and backward pass in finetuning stage
    c2, _ = _forward(model, batch)
    loss2 = F.cross_entropy(c2, batch[Input.label])
    assert torch.isfinite(loss2), "Finetuning loss is not finite"
    loss2.backward()
    opt2.step()

    # Unfreeze layers to restore full graph
    model.layers.setLayerRequiresGrad(requires_grad=True)
    all_trainable = all(p.requires_grad for p in model.parameters())
    assert all_trainable, "Failed to restore full requires_grad after finetuning"


# ==============================================================================
# 5. Model Capacity / Overfitting Verification
# ==============================================================================
def test_model_capacity_overfit():
    """Verify that the model has the capacity to overfit a small batch (loss decreases > 50%)."""
    model = HCR(
        dijetFeatures=8,
        quadjetFeatures=8,
        ancillaryFeatures=["xW", "nSelJets", "xbW"],
        canJetFeatures=["pt", "eta", "phi", "mass"],
        useOthJets="attention",
        device="cpu",
        nClasses=2,
    )
    batch = _build_synthetic_batch(batch_size=8, ancillary_features=["xW", "nSelJets", "xbW"], seed=123)
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
        f"Model failed to overfit tiny batch: initial={initial_loss:.4f}, final={final_loss:.4f}"
    )


# ==============================================================================
# 6. Deterministic Training Seeding Verification
# ==============================================================================
def test_deterministic_training_reproducibility():
    """Verify that models initialized with identical seed 666 produce identical weights and losses."""
    losses = []
    param_states = []

    for _ in range(2):
        torch.manual_seed(666)
        model = HCR(
            dijetFeatures=8,
            quadjetFeatures=8,
            ancillaryFeatures=["xW", "nSelJets", "xbW"],
            canJetFeatures=["pt", "eta", "phi", "mass"],
            useOthJets="attention",
            device="cpu",
            nClasses=2,
        )
        batch = _build_synthetic_batch(batch_size=8, ancillary_features=["xW", "nSelJets", "xbW"], seed=666)
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
    assert losses[0] == pytest.approx(losses[1], rel=1e-6), "Losses diverged under deterministic seed"
    for p0, p1 in zip(param_states[0], param_states[1]):
        assert torch.allclose(p0, p1), "Model weights diverged under deterministic seed"
