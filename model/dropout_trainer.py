import numpy as np

import torch
from tqdm import tqdm
import model.trainer_base as trainer_base
from torch import Tensor


def crps_ensemble(
    truth,  # TRUTH
    predicted,  # FORECAST
    weights = None,
    dim = (),
    reduction="mean",
) -> Tensor:
    """
    .. Author: Salva Rühling Cachay

    pytorch adaptation of https://github.com/TheClimateCorporation/properscoring/blob/master/properscoring/_crps.py#L187
    but implementing the fair, unbiased CRPS as in Zamo & Naveau (2018; https://doi.org/10.1007/s11004-017-9709-7)

    This implementation is based on the identity:
    .. math::
        CRPS(F, x) = E_F|X - x| - 1/2 * E_F|X - X'|
    where X and X' denote independent random variables drawn from the forecast
    distribution F, and E_F denotes the expectation value under F.

    We use the fair, unbiased formulation of the ensemble CRPS, which is particularly important for small ensembles.
    Anecdotically, the unbiased CRPS leads to slightly smaller (i.e. "better") values than the biased version.
    Basically, we use n_members * (n_members - 1) instead of n_members**2 to average over the ensemble spread.
    See Zamo & Naveau (2018; https://doi.org/10.1007/s11004-017-9709-7) for details.

    Alternative implementation: https://github.com/NVIDIA/modulus/pull/577/files
    """
    assert truth.ndim == predicted.ndim - 1, f"{truth.shape=}, {predicted.shape=}"
    assert truth.shape == predicted.shape[1:]  # ensemble ~ first axis
    n_members = predicted.shape[0]
    skill = (predicted - truth).abs().mean(dim=0)
    # insert new axes so forecasts_diff expands with the array broadcasting
    # torch.unsqueeze(predictions, 0) has shape (1, E, ...)
    # torch.unsqueeze(predictions, 1) has shape (E, 1, ...)
    forecasts_diff = torch.unsqueeze(predicted, 0) - torch.unsqueeze(predicted, 1)
    # Forecasts_diff has shape (E, E, ...)
    # Old version: score += - 0.5 * forecasts_diff.abs().mean(dim=(0, 1))
    # Using n_members * (n_members - 1) instead of n_members**2 is the fair, unbiased CRPS. Better for small ensembles.
    spread = forecasts_diff.abs().sum(dim=(0, 1)) / (n_members * (n_members - 1))
    crps = skill - 0.5 * spread
    # score has shape (...)  (same as observations)
    if reduction == "none":
        return crps
    assert reduction == "mean", f"Unknown reduction {reduction}"
    if weights is not None:  # weighted mean
        crps = (crps * weights).sum(dim=dim) / weights.expand(crps.shape).sum(dim=dim)
    else:
        crps = crps.mean(dim=dim)
    return crps

def foot_slide_one_step_relative_to_gt(
    *,
    last_x,      # (B,C)
    pred_x,      # (B,C)
    next_x,      # (B,C)
    dataset,
    mode,
    foot_idx,    # list/array of indices (len can be 1/2/4/...)
):
    contact_threshold = 0.3

    device = pred_x.device
    B = pred_x.shape[0]

    last_np = last_x.detach().cpu().numpy()
    pred_np = pred_x.detach().cpu().numpy()
    next_np = next_x.detach().cpu().numpy()

    foot_idx = np.asarray(foot_idx, dtype=np.int64)
    if foot_idx.ndim != 1 or foot_idx.size < 1:
        raise ValueError(f"foot_idx must be a 1D non-empty list/array, got {foot_idx}")

    losses = []

    for b in range(B):
        last_j = dataset.x_to_jnts(dataset.denorm_data(last_np[b].copy()), mode=mode)
        pred_j = dataset.x_to_jnts(dataset.denorm_data(pred_np[b].copy()), mode=mode)
        next_j = dataset.x_to_jnts(dataset.denorm_data(next_np[b].copy()), mode=mode)

        # Accept either (J,3) or (T,J,3) with T=1
        if last_j.ndim == 3:
            last_j = last_j[-1]
        if pred_j.ndim == 3:
            pred_j = pred_j[-1]
        if next_j.ndim == 3:
            next_j = next_j[-1]

        # (F,3)
        last_f = last_j[foot_idx]
        pred_f = pred_j[foot_idx]
        next_f = next_j[foot_idx]

        # One-step displacement magnitudes in xz: (F,)
        d_pred = pred_f - last_f
        d_gt   = next_f - last_f
        dxz_pred = np.linalg.norm(d_pred[:, [0, 2]], axis=-1)
        dxz_gt   = np.linalg.norm(d_gt[:,   [0, 2]], axis=-1)

        # Height (y) at next step for contact gating: (F,)
        y = next_f[:, 1]

        F = dxz_pred.shape[0]

        # ---- Handle arbitrary F safely ----
        # If F>=4: keep your original convention (use joints 1 and 3, and contact by max over each foot’s 2 joints)
        # If F==2: treat as one joint per foot (2 feet), contact per joint
        # If F==1: single foot joint, contact scalar
        if F >= 4:
            # use indices 1 and 3 if they exist
            pair = [1, 3]
            dxz_pred_pair = dxz_pred[pair]
            dxz_gt_pair   = dxz_gt[pair]

            # contact: group into 2 feet by taking max over halves (no assumption of exactly 2 joints/foot beyond ">=4")
            # use first half vs second half
            half = F // 2
            y_max = np.array([np.max(y[:half]), np.max(y[half:])], dtype=np.float32)

            ratio = np.clip(y_max / contact_threshold, 0.0, 1.0)
            factor = 2.0 - 2.0 ** ratio  # (2,)

            extra = np.maximum(dxz_pred_pair - dxz_gt_pair, 0.0)  # (2,)
            slide = float(np.mean(extra * factor))

        elif F == 2:
            # one joint per foot
            ratio = np.clip(y / contact_threshold, 0.0, 1.0)      # (2,)
            factor = 2.0 - 2.0 ** ratio                            # (2,)
            extra = np.maximum(dxz_pred - dxz_gt, 0.0)             # (2,)
            slide = float(np.mean(extra * factor))

        else:  # F == 1
            ratio = float(np.clip(y[0] / contact_threshold, 0.0, 1.0))
            factor = float(2.0 - 2.0 ** ratio)
            extra = float(max(dxz_pred[0] - dxz_gt[0], 0.0))
            slide = extra * factor

        losses.append(slide)

    return torch.tensor(losses, device=device, dtype=pred_x.dtype).mean()


def crps_plus_one_step_footslide_loss(
    *,
    truth: torch.Tensor,        # (B,C)
    predicted: torch.Tensor,    # (E,B,C)
    last_x: torch.Tensor,       # (B,C)
    dataset,
    foot_idx,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined loss = CRPS + alpha * foot_slide

    No new hyperparameters:
      alpha is set automatically to match magnitudes:
        alpha = CRPS.detach() / (foot_slide.detach() + eps)
    """
    # CRPS (scalar)
    crps = crps_ensemble(truth=truth, predicted=predicted, dim=())

    # foot slide per ensemble member (scalar), averaged across E
    slide_terms = []
    for e in range(predicted.shape[0]):
        slide_terms.append(
            foot_slide_one_step_relative_to_gt(
                last_x=last_x,
                pred_x=predicted[e],
                next_x=truth,
                dataset=dataset,
                mode=dataset.data_component[0],
                foot_idx=foot_idx,
            )
        )
    slide = torch.stack(slide_terms).mean()

    # auto-balance (no tunable lambda)
    eps = 1e-8
    alpha = crps.detach() / (slide.detach() + eps)
    loss = crps + alpha * slide

    return loss, crps, slide

class DropoutTrainer(trainer_base.BaseTrainer):
    NAME = 'MCDropout'
    def __init__(self, config, dataset, device):
        super(DropoutTrainer, self).__init__(config, dataset, device)
        optimizer_config = config['optimizer']
        self.loss_type = optimizer_config.get("loss_type", "l1")
        self.use_all_pairs = optimizer_config.get('use_all_pairs', False)
        self.crps_rollout_steps = optimizer_config.get("crps_rollout_steps", 2)
        self.crps_horizon_weights = optimizer_config.get("crps_horizon_weights", None)

    def _combine_crps_losses(self, losses):
        """
        losses: list of scalar tensors, one per rollout horizon
        """
        if self.crps_horizon_weights is None:
            return torch.stack(losses).mean()

        weights = torch.as_tensor(
            self.crps_horizon_weights,
            dtype=losses[0].dtype,
            device=losses[0].device,
        )

        if weights.numel() != len(losses):
            raise ValueError(
                f"crps_horizon_weights has length {weights.numel()}, "
                f"but rollout produced {len(losses)} losses."
            )

        weights = weights / weights.sum()
        return (torch.stack(losses) * weights).sum()

    def compute_loss(self, model, last_x, next_x, future_x=None):
        if self.loss_type == 'l1' or self.loss_type == 'l2':
            pred_x = model(last_x)
    
            if self.loss_type == 'l1':
                loss_diff = torch.nn.functional.l1_loss(pred_x, next_x)
    
            elif self.loss_type == 'l2':
                loss_diff = torch.nn.functional.mse_loss(pred_x, next_x)
        elif self.loss_type == 'crps':
            H = self.crps_rollout_steps
            E = self.config["optimizer"]["crps_ens"]

            if future_x is None:
                raise ValueError("CRPS requires future_x with shape (B, H, D).")

            if future_x.shape[1] != H:
                raise ValueError(
                    f"future_x has horizon {future_x.shape[1]}, expected {H}."
                )

            B = last_x.shape[0]

            # Duplicate the batch once, so each MC sample is in the same large batch.
            # Shape: (E, B, D) -> (E*B, D)
            x_rep = (
                last_x.unsqueeze(0)
                .expand(E, *last_x.shape)
                .reshape(E * B, *last_x.shape[1:])
            )

            losses = []
            pred_x = None

            for h in range(H):
                # One forward pass for all MC samples at this rollout step
                pred_rep = model(x_rep)  # (E*B, D)

                # Back to ensemble layout: (E, B, D)
                predicted = pred_rep.reshape(E, B, *pred_rep.shape[1:])

                truth_h = future_x[:, h, ...]  # (B, D)

                losses.append(
                    crps_ensemble(
                        truth=truth_h,
                        predicted=predicted,
                        dim=(),
                    )
                )

                # logging prediction = ensemble mean at current horizon
                pred_x = predicted.mean(dim=0)

                # autoregressive rollout: each sample feeds its own next step
                x_rep = pred_rep

            loss_diff = self._combine_crps_losses(losses)
            return loss_diff, pred_x
        elif self.loss_type == 'crps-foot-slide':
            # ensemble
            preds = [model(last_x) for _ in range(2)]
            predicted = torch.stack(preds, dim=0)  # (E,B,C)
    
            # required existing fields:
            #   self.foot_idx : foot joint indices in your skeleton layout (len typically 4)
            #   self.mode0    : the mode you already pass to x_to_jnts
            loss, crps, slide = crps_plus_one_step_footslide_loss(
                truth=next_x,
                predicted=predicted,
                last_x=last_x,
                dataset=self.dataset,
                foot_idx=self.dataset.foot_idx,
            )
    
            pred_x = predicted.mean(dim=0)
            return loss, pred_x
              
        return loss_diff, pred_x #.detach() 

    def train_loop(self, ep, model):
        ep_loss_sum = 0.0
        num_batches = 0

        self._update_lr_schedule(self.optimizer, ep - 1)

        model.train()  # dropout enabled during training
        pbar = tqdm(self.train_dataloader, colour="green")

        for frames in pbar:
            frames = frames.to(self.device).float()

            if frames.dim() != 3:
                raise ValueError(f"Expected frames shaped (B,T,D). Got {tuple(frames.shape)}")

            B, T, D = frames.shape

            if self.loss_type == 'crps':
                H = self.crps_rollout_steps

                if T < H + 1:
                    continue

                if self.use_all_pairs:
                    # start states: t = 0 .. T-H-1
                    # xcur shape: (B, W, D), where W = T - H
                    xcur = frames[:, :T - H, :]  # (B, W, D)

                    # future targets stacked as (B, W, H, D)
                    future = torch.stack(
                        [frames[:, h:T - H + h, :] for h in range(1, H + 1)],
                        dim=2
                    )

                    # flatten windows into batch
                    xcur = xcur.reshape(-1, D)        # (B*W, D)
                    future = future.reshape(-1, H, D) # (B*W, H, D)
                else:
                    xcur = frames[:, 0, :]            # (B, D)
                    future = frames[:, 1:H + 1, :]    # (B, H, D)

                xnext = future[:, 0, :]  # first-step target, kept for API compatibility

            else:
                if T < 2:
                    continue

                if self.use_all_pairs:
                    xcur = frames[:, :-1, :].reshape(-1, D)
                    xnext = frames[:, 1:, :].reshape(-1, D)
                else:
                    xcur = frames[:, 0, :]
                    xnext = frames[:, 1, :]

                future = None

            self.optimizer.zero_grad(set_to_none=True)

            loss, pred = self.compute_loss(model, xcur, xnext, future_x=future)

            loss.backward()
            self.optimizer.step()

            if hasattr(model, "update") and callable(model.update):
                model.update()

            ep_loss_sum += float(loss.item())
            num_batches += 1

            pbar.set_description(f"ep:{ep}, loss:{loss.item():.4f}")

        avg_loss = ep_loss_sum / max(1, num_batches)

        train_info = {
            "epoch": ep,
            "loss": avg_loss,
        }
        return train_info
