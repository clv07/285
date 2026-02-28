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

class DropoutTrainer(trainer_base.BaseTrainer):
    NAME = 'MCDropout'
    def __init__(self, config, dataset, device):
        super(DropoutTrainer, self).__init__(config, dataset, device)
        optimizer_config = config['optimizer']
        self.loss_type = optimizer_config.get("loss_type", "l1")
        self.use_all_pairs = optimizer_config.get('use_all_pairs', False)

    def compute_loss(self, model, last_x, next_x):
        if self.loss_type == 'l1' or self.loss_type == 'l2':
            pred_x = model(last_x)
    
            if self.loss_type == 'l1':
                loss_diff = torch.nn.functional.l1_loss(pred_x, next_x)
    
            elif self.loss_type == 'l2':
                #loss_diff = torch.sum(torch.square(target - estimated), dim=-1).mean()
                loss_diff = torch.nn.functional.mse_loss(pred_x, next_x)
        elif self.loss_type == 'crps':
            preds = []
            for _ in range(2):
                preds.append(model(last_x))  # (B, C, ...)
    
            predicted = torch.stack(preds, dim=0)  # (E, B, C, ...)
            truth = next_x                          # (B, C, ...)
    
            # If you want a scalar loss:
            loss_diff = crps_ensemble(truth=truth, predicted=predicted, dim=())  # mean over all dims
    
            # You still need to return a single pred for logging; use ensemble mean:
            pred_x = predicted.mean(dim=0)  # (B, C, ...)
              
        return loss_diff, pred_x #.detach() 

    def train_loop(self, ep, model):
        ep_loss_sum = 0.0
        num_batches = 0

        self._update_lr_schedule(self.optimizer, ep - 1)

        model.train()  # dropout enabled during training
        pbar = tqdm(self.train_dataloader, colour="green")

        for frames in pbar:
            frames = frames.to(self.device).float()

            # Expect frames shaped (B, T, D) or (B, D)
            if frames.dim() == 3:
                B, T, D = frames.shape
                if T < 2:
                    continue

                if self.use_all_pairs:
                    xcur = frames[:, :-1, :].reshape(-1, D)
                    xnext = frames[:, 1:, :].reshape(-1, D)
                else:
                    xcur = frames[:, 0, :]
                    xnext = frames[:, 1, :]
            elif frames.dim() == 2:
                # If dataloader already yields (B, D) pairs, this trainer can't infer xnext.
                # Keep behavior explicit to avoid silent mistakes.
                raise ValueError(
                    "Expected frames shaped (B,T,D) with T>=2. Got (B,D). "
                    "Update dataloader to return sequences or modify trainer to accept paired batches."
                )
            else:
                raise ValueError(f"Unexpected frames shape: {tuple(frames.shape)}")

            self.optimizer.zero_grad(set_to_none=True)

            loss, pred = self.compute_loss(model, xcur, xnext)

            loss.backward()
            self.optimizer.step()

            # If your model implements EMA update via model.update(), keep it compatible:
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
