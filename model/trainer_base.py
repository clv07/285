import abc
import copy
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import torch
import wandb
import torch.optim as optim

from torch.utils.data import DataLoader

import util.vis_util as vis_util
import util.logging as logging_util
import util.save as save_util
from util.eval import compute_test_metrics, compute_long_test_metrics
import yaml
import os
import random

from model import model_builder      # needed to rebuild model

import numpy as np

def save_a_checkpoint(path, model, optimizer, epoch, extra=None):
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "extra": extra or {},
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)

def load_checkpoint(path, model, optimizer=None, device="cpu", strict=True):
    print(f"loading checkpoint from {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=strict)

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    # Restore RNG (optional but good for exact reproducibility)
    """
    if "rng" in ckpt:
        import random
        random.setstate(ckpt["rng"]["python"])
        np.random.set_state(ckpt["rng"]["numpy"])
        torch.set_rng_state(ckpt["rng"]["torch"])
        if torch.cuda.is_available() and ckpt["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])"""

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    extra = ckpt.get("extra", {})
    return start_epoch, extra


def _crps_ensemble_np(samples: np.ndarray, y: float = 0.0) -> np.ndarray:
    """
    Empirical CRPS for an ensemble.

    samples: (..., K)
    y: scalar observation (use y=0 if samples are errors)
    returns: (...) CRPS

    CRPS = E|X-y| - 0.5 E|X-X'|
    """
    samples = np.asarray(samples)
    term1 = np.mean(np.abs(samples - y), axis=-1)  # (...,)

    diffs = np.abs(samples[..., :, None] - samples[..., None, :])  # (...,K,K)
    term2 = 0.5 * np.mean(diffs, axis=(-1, -2))                    # (...,)

    return term1 - term2


def _per_timestep_jointpos_err_np(output_jnts: np.ndarray, ref_jnts: np.ndarray) -> np.ndarray:
    """
    output_jnts: (K,T,J,3)
    ref_jnts:    (T,J,3)
    returns:     (T,K) where entry (t,k) is mean_j ||pred-ref||_2 at timestep t
    """
    diff = output_jnts - ref_jnts[None, ...]        # (K,T,J,3)
    dist = np.linalg.norm(diff, axis=-1)            # (K,T,J)
    err = dist.mean(axis=-1)                        # (K,T)
    return np.transpose(err, (1, 0))                # (T,K)


def update_val_crps_running(
    crps_sum: dict,
    crps_cnt: dict,
    output_jnts: np.ndarray,
    ref_jnts: np.ndarray,
    step: int = 10,
    prefix: str = "val",
):
    """
    Accumulate CRPS at timesteps 0, step, 2*step, ... for a single clip.

    crps_sum/crps_cnt: running accumulators keyed by f"{prefix}/T{t}/CRPS"
    output_jnts: (K,T,J,3)
    ref_jnts:    (T,J,3)
    """
    err_tk = _per_timestep_jointpos_err_np(output_jnts, ref_jnts)  # (T,K)
    T = err_tk.shape[0]

    for t in range(0, T, step):
        v = float(_crps_ensemble_np(err_tk[t], y=0.0))
        if np.isfinite(v):
            key = f"{prefix}/T{t}/CRPS"
            crps_sum[key] = crps_sum.get(key, 0.0) + v
            crps_cnt[key] = crps_cnt.get(key, 0.0) + 1.0


def finalize_crps_log(crps_sum: dict, crps_cnt: dict) -> dict:
    """
    Convert running sums/counts into mean CRPS values ready for logger.log_epoch().
    """
    out = {}
    for k, s in crps_sum.items():
        c = crps_cnt.get(k, 0.0)
        if c > 0:
            out[k] = s / c
    return out


def finalize_crps_pairs(crps_sum: dict, crps_cnt: dict):
    """
    Returns list of (lead_time, mean_crps)
    """
    pairs = []

    for k, s in crps_sum.items():
        c = crps_cnt.get(k, 0.0)
        if c <= 0:
            continue

        # key format: "val/T{t}/CRPS"
        # extract t
        t_str = k.split("/")[1]  # "T{t}"
        t = int(t_str[1:])

        pairs.append((t, s / c))

    pairs.sort(key=lambda x: x[0])
    return pairs


class BaseTrainer():
    def __init__(self, config, dataset, device):
        self.config = config
        self.device = device
        self.dataset = dataset

        optimizer_config = config['optimizer']
        self.batch_size = optimizer_config['mini_batch_size']
        self.num_rollout = optimizer_config['rollout']
        self.initial_lr = optimizer_config['initial_lr']
        self.final_lr = optimizer_config['final_lr']
        self.peak_student_rate = optimizer_config.get('peak_student_rate',1.0)
        self._get_schedule_samp_routines(config['optimizer'])
        
        test_config = config['test']
        self.test_interval = test_config["test_interval"]
        self.test_num_steps = test_config["test_num_steps"]
        self.test_num_trials = test_config["test_num_trials"]
        self.plot_every_n_test=test_config["plot_every_n_test"]
        self.save_checkpoint_every_n_test=test_config["save_checkpoint_every_n_test"]
        self.plot_indices=test_config["plot_indices"]
        self.crps_step=test_config["crps_step"]
        self.max_plot_traj=test_config["max_plot_traj"]
        
        self.frame_dim = dataset.frame_dim
        self.train_dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self.logger =  logging_util.wandbLogger(proj_name="{}_{}".format(self.NAME, dataset.NAME), run_name=self.NAME, config=config)

        self.plot_jnts_fn = self.dataset.plot_jnts if hasattr(self.dataset, 'plot_jnts') and callable(self.dataset.plot_jnts) \
                                                        else vis_util.vis_skel

        self.plot_traj_fn = self.dataset.plot_traj if hasattr(self.dataset, 'plot_traj') and callable(self.dataset.plot_traj) \
                                                        else vis_util.vis_traj
        return

    @abc.abstractmethod
    def train_loop(self, model):
        return    

    def _init_optimizer(self, model):
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=self.initial_lr)

    def _update_lr_schedule(self, optimizer, epoch):
        """Decreases the learning rate linearly"""
        lr = self.initial_lr - (self.initial_lr - self.final_lr) * epoch / float(self.total_epochs)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def _get_schedule_samp_routines(self, optimizer_config):
        if 'total_epochs' in optimizer_config.keys():
            self.total_epochs = optimizer_config['total_epochs']
        else:
            self.anneal_times = optimizer_config['anneal_times']
            self.initial_teacher_epochs = optimizer_config.get('initial_teacher_epochs',1)
            self.end_teacher_epochs = optimizer_config.get('end_teacher_epochs',1)
            self.teacher_epochs = optimizer_config['teacher_epochs']
            self.ramping_epochs = optimizer_config['ramping_epochs']
            self.student_epochs = optimizer_config['student_epochs']
            self.use_schedule_samp = self.ramping_epochs != 0 or self.student_epochs != 0
            
            self.initial_schedule = torch.zeros(self.initial_teacher_epochs)
            self.end_schedule = torch.zeros(self.end_teacher_epochs)
            self.sample_schedule = torch.cat([ 
                    # First part is pure teacher forcing
                    torch.zeros(self.teacher_epochs),
                    # Second part with schedule sampling
                    torch.linspace(0.0, self.peak_student_rate, self.ramping_epochs),
                    # last part is pure student
                    torch.ones(self.student_epochs) * self.peak_student_rate,
    
            ])
            self.sample_schedule = torch.cat([self.sample_schedule  for _ in range(self.anneal_times)], axis=-1)
            self.sample_schedule = torch.cat([self.initial_schedule, self.sample_schedule, self.end_schedule])
           
            self.total_epochs = self.sample_schedule.shape[0]


    def train_model(self, model, out_model_file, int_output_dir, log_file, resume_path=None):
        self._init_optimizer(model)
        start_ep = 0

        if resume_path is not None:
            assert os.path.exists(resume_path)

        if resume_path is not None and os.path.exists(resume_path):
            start_ep, extra = load_checkpoint(
                resume_path, model, optimizer=self.optimizer, device=self.device, strict=True
            )
        
        for ep in range(start_ep, self.total_epochs):
            loss_stats = self.train_loop(ep, model)
            if ep == 0:
                continue
            if ep % self.test_interval == 0:
                num_nans = self.evaluate(ep, model, int_output_dir, out_model_file)
                # save_util.save_weight(model, int_output_dir+'_ep{}.pth'.format(ep))
                # save_util.save_weight(model, out_model_file)
                
            self.logger.log_epoch(loss_stats)
            self.logger.print_log(loss_stats)
            
        save_util.save_weight(model, out_model_file)

    def evaluate(self, ep, model, result_ouput_dir, out_model_file=""):
        model.eval()
        NaN_clip_num = 0
    
        stats_dict = defaultdict(float)
        long_stats_dict = defaultdict(float)
    
        mode0 = self.dataset.data_component[0]
        n_total = len(self.dataset.test_valid_idx)
    
        # Tune these
        B_eval = 3000                # clips per batch
        singular_gif = self.config['test']['single_test']
        #do_long = False             # turn on sparingly
        long_T = 100
        long_K = 1
        crps_sum = {}
        crps_cnt = {}
        do_plot = (ep - self.test_interval) % (self.test_interval * self.plot_every_n_test) == 0 or ep == -1
        save_checkpoint = (ep - self.test_interval) % (self.test_interval * self.save_checkpoint_every_n_test) == 0

        plot_idx = set(self.plot_indices)
        plot_clips = self.max_plot_traj
    
        with torch.inference_mode():
            for b0 in range(0, n_total, B_eval):
                b1 = min(b0 + B_eval, n_total)
                B = b1 - b0
    
                st_idx_batch = self.dataset.test_valid_idx[b0:b1]
                ref_clip_batch = self.dataset.test_ref_clips[b0:b1]
    
                # start_x: (B, D)
                start_x = torch.from_numpy(np.stack([rc[0] for rc in ref_clip_batch], axis=0)).float().to(self.device)
    
                # x: (B, K, T, D)
                x = model.eval_seq(start_x, None, self.test_num_steps, self.test_num_trials)
                #x_long = model.eval_seq(start_x, None, long_T, long_K)  # (1, Klong, Tlong, D)
    
                # NaN accounting per-clip
                bad_clip = torch.isnan(x).flatten(start_dim=1).any(dim=1)  # (B,)
                NaN_clip_num += int(bad_clip.sum().item())
    
                x_np = x.detach().cpu().numpy()  # (B,K,T,D)
                #x_long_np = x_long.detach().cpu().numpy()
    
                for bi in tqdm(range(B)):
                    st_idx = st_idx_batch[bi]
                    ref_clip = ref_clip_batch[bi]  # (T,D)
    
                    # GT joints once
                    ref_jnts = self.dataset.x_to_jnts(
                        self.dataset.denorm_data(ref_clip.copy()),
                        mode=mode0
                    )  # (T,J,3)
    
                    # Denorm all trials at once: (K,T,D)
                    den = self.dataset.denorm_data(x_np[bi])  # uses numpy broadcasting
    
                    # Batched joints: (K,T,J,3)
                    output_jnts = self.dataset.x_to_jnts_batched(den, mode=mode0)

                    update_val_crps_running(
                        crps_sum, crps_cnt,
                        output_jnts=output_jnts,
                        ref_jnts=ref_jnts,
                        step=self.crps_step,
                        prefix="val"
                    )
    
                    stats = compute_test_metrics(
                        links=self.dataset.links,
                        foot_idx=self.dataset.foot_idx,
                        output_jnts=output_jnts,
                        ref_jnts=ref_jnts,
                    )
    
                    for k, v in stats.items():
                        stats_dict[k] += float(v)
                    if do_plot and bi in plot_idx:
                        # 1) GT joint plot (same behavior as old code)
                        clip_dir = os.path.join(result_ouput_dir, str(st_idx))
                        os.makedirs(clip_dir, exist_ok=True)
                        epoch_dir = os.path.join(clip_dir, f"epoch_{ep}")
                        os.makedirs(epoch_dir, exist_ok=True)
                        if ep <= self.test_interval +1 and not singular_gif:
                            self.plot_jnts_fn(ref_jnts[None, ...], f"{clip_dir}/gt")  # (1,T,J,3)

                        if singular_gif:
                            print("SINGULAR GIF")
                            K = output_jnts.shape[0]
                            for ki in range(K):
                                # Save each trial alone; shape expected: (1,T,J,3)
                                trial_jnts = output_jnts[ki][None, ...]
                                self.plot_jnts_fn(trial_jnts, f"{epoch_dir}/trial_{ki:03d}_jnts")
    
                        # Plot GT + capped predicted trials together (mode0 only)
                        if not bool(bad_clip[bi].item()) and not singular_gif:
                            K_plot = min(output_jnts.shape[0], plot_clips)
                            out_plot = output_jnts[:K_plot]  # (K_plot,T,J,3)
                    
                            jnts_all = np.concatenate([ref_jnts[None, ...], out_plot], axis=0)  # (K_plot+1,T,J,3)
                            self.plot_jnts_fn(jnts_all, f"{epoch_dir}/gt_plus_{K_plot}_trials_jnts")
                    
                            # Optional trajectory plot with same capped set
                            self.plot_traj_fn(jnts_all, f"{epoch_dir}/traj_{K_plot}")
    
                        # 3) Trajectory plot with GT + all generated trials (mode0), like old plot_traj_fn
                        #    Shape expected: (Ncurves, T, J, 3)
                        traj_all = [ref_jnts] + [output_jnts[ki] for ki in range(output_jnts.shape[0])]
                        self.plot_traj_fn(np.array(traj_all), f"{epoch_dir}/traj")
    
                        # 4) GT-only trajectory plot saved alongside
                        self.plot_traj_fn(np.array([ref_jnts]), f"{clip_dir}/gt")
                    # -------------------------
    
                    # Optional long-horizon
                    # if do_long:

                    """
                    den_long = self.dataset.denorm_data(x_long_np[bi])
                    out_long = self.dataset.x_to_jnts_batched(den_long, mode=mode0)

                    long_stats = compute_long_test_metrics(
                        links=self.dataset.links,
                        foot_idx=self.dataset.foot_idx,
                        output_jnts=out_long,
                        ref_jnts=ref_jnts,
                    )
                    for k, v in long_stats.items():
                        long_stats_dict[k] += float(v)
                    """
    
        # Average + log
       
        
        n = float(n_total)
        stats_dict = {k: v / n for k, v in stats_dict.items()}
        self.logger.log_epoch(stats_dict)
        if ep == -1:
            crps_pairs = finalize_crps_pairs(crps_sum, crps_cnt)
        
            for t, mean_crps in crps_pairs:
                wandb.log(
                    {
                        "val/lead_time": t,
                        "val/CRPS": mean_crps,
                        "epoch": ep,
                    }
                )
        else:
            self.logger.log_epoch(finalize_crps_log(crps_sum, crps_cnt))

        if save_checkpoint:
            print("saving model checkpoint")
            # save_util.save_weight(model, result_ouput_dir+'_ep{}.pth'.format(ep))
            save_a_checkpoint(
                path=os.path.join(result_ouput_dir, f"latest.pth"),
                model=model,
                optimizer=self.optimizer,
                epoch=ep,
            )
        """
        # if do_long:
        long_stats_dict = {f"long/{k}": (v / n) for k, v in long_stats_dict.items()}
        self.logger.log_epoch(long_stats_dict)
        """

        print(NaN_clip_num)
        return NaN_clip_num