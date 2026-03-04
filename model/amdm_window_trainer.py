import numpy as np

import torch
from tqdm import tqdm
import model.trainer_base as trainer_base

class AMDMTrainer(trainer_base.BaseTrainer):
    NAME = 'AMDM'
    def __init__(self, config, dataset, device):
        super(AMDMTrainer, self).__init__(config, dataset, device)
        optimizer_config = config['optimizer']
        self.full_T = optimizer_config.get('full_T', False)
        self.consistency_on = optimizer_config.get('consistency_on', False)
        self.consist_loss_weight = optimizer_config.get('consist_loss_weight',1)
        self.loss_type = config["diffusion"]["loss_type"] 
        self.recon_on = optimizer_config.get('recon_on', False)
        self.recon_loss_weight = optimizer_config.get('recon_loss_weight', 1)
        self.diffusion_loss_weight = optimizer_config.get('diffusion_loss_weight', 1)
        self.detach_step = optimizer_config.get('detach_step',3)
        self.window_size = config["diffusion"]["window_size"]    

    def compute_teacher_loss(self, model, sampled_frames, extra_info):
        #st_index = random.randint(0,sampled_frames.shape[1]-2)
        #print('teacher forcing')
        K = self.window_size
        last_frame = sampled_frames[:,0,:]
        ground_truth = sampled_frames[:,1:1+K,:]
            
        self.optimizer.zero_grad()
        diff_loss, pred_frame = model.compute_loss(last_frame,  ground_truth, None, extra_info)
        loss = self.diffusion_loss_weight * diff_loss 
    
        loss.backward()
        self.optimizer.step()
        model.update()

        return {"diff_loss":diff_loss.item()}
    

    def compute_student_loss(self, model, sampled_frames, sch_samp_prob, extra_info):
        #print('student forcing')
        loss_diff_sum, loss_consist_sum = 0, 0
        
        batch_size = sampled_frames.shape[0]
        shrinked_batch_size = batch_size//model.T
        pred_window = None
        for st_index in range(self.num_rollout -1):
            self.optimizer.zero_grad()
            next_index = st_index + 1
            ground_truth = sampled_frames[:,next_index,:]


            if self.full_T:
                Kwin = self.window_size
                shrinked_batch_size = batch_size  # keep naming consistent
            
                # Build window GT for this rollout step
                gt_start = st_index + 1
                gt_end = gt_start + Kwin
                ground_truth_win = sampled_frames[:, gt_start:gt_end, :]  # (B, Kwin, D) if enough frames
            
                # If not enough frames, stop rollout
                if ground_truth_win.shape[1] < Kwin:
                    break
            
                # Choose last_frame (B, D)
                if st_index == 0:
                    last_frame = sampled_frames[:, 0, :]
                else:
                    if pred_window is None:
                        last_frame = sampled_frames[:, st_index, :]
                    else:
                        last_frame = pred_window.detach()[:, 0, :]  # use first frame of predicted window
            
                    teacher_forcing_mask = torch.bernoulli(
                        1.0 - torch.ones(batch_size, device=last_frame.device) * sch_samp_prob
                    ).bool()
                    last_frame[teacher_forcing_mask] = sampled_frames[teacher_forcing_mask, st_index, :]
            
                # Expand last_frame across diffusion timesteps -> (B*T, D)
                last_frame_expanded = last_frame[:, None, :].expand(-1, model.T, -1).reshape(batch_size * model.T, -1)
            
                # Expand window GT across diffusion timesteps -> (B*T, Kwin, D)
                ground_truth_expanded = ground_truth_win[:, None, :, :].expand(-1, model.T, -1, -1).reshape(
                    batch_size * model.T, Kwin, -1
                )
            
                # Build per-sample ts indices -> (B*T,)
                ts = torch.arange(0, model.T, device=self.device)
                ts = ts[None, ...].expand(batch_size, -1).reshape(-1)
            
                # (Optional but recommended) teacher loss: condition on true previous frame
                teacher_last = sampled_frames[:, st_index, :]
                teacher_last_expanded = teacher_last[:, None, :].expand(-1, model.T, -1).reshape(batch_size * model.T, -1)
                diff_loss_teacher, _ = model.compute_loss(teacher_last_expanded, ground_truth_expanded, ts, extra_info)
            
                # Student loss: condition on mixed (teacher/student) frame
                diff_loss_student, pred_x0 = model.compute_loss(last_frame_expanded, ground_truth_expanded, ts, extra_info)
            
                # pred_x0 is (B*T, Kwin, D) -> reshape back to (B, T, Kwin, D)
                pred_x0 = pred_x0.reshape(batch_size, model.T, Kwin, -1)
            
                # Choose which diffusion step’s prediction to roll forward with.
                # Using ts=0 slice is consistent with your original full_T logic.
                pred_window = pred_x0[:, 0, :, :]  # (B, Kwin, D)
            
                diff_loss = diff_loss_student + diff_loss_teacher
                loss = self.diffusion_loss_weight * diff_loss
            
            # if self.full_T:
            #     shrinked_batch_size = batch_size
            #     if st_index == 0:
            #         last_frame = sampled_frames[:,0,:]
            #         last_frame_expanded = last_frame[:,None,:].expand(-1, model.T, -1).reshape(shrinked_batch_size*model.T, -1)
            #     else:
            #         last_frame = pred_frame.detach().reshape(shrinked_batch_size, model.T, -1)[:,0,:]
            #         teacher_forcing_mask = torch.bernoulli(1.0-torch.ones(shrinked_batch_size, device=pred_frame.device) *sch_samp_prob).bool()
            #         last_frame[teacher_forcing_mask] = sampled_frames[teacher_forcing_mask, st_index, :]
            #         last_frame_expanded = last_frame[:,None,:].expand(-1, model.T, -1).reshape(shrinked_batch_size*model.T, -1)
                
            #         teacher_forcing_mask = torch.bernoulli(
            #             1.0 - torch.ones(batch_size, device=last_frame.device) * sch_samp_prob
            #         ).bool()
            #         last_frame[teacher_forcing_mask] = sampled_frames[teacher_forcing_mask, st_index, :]

            #     ground_truth_expanded = ground_truth[:,None,:].expand(-1, model.T, -1).reshape(shrinked_batch_size*model.T, -1)

            #     ts = torch.arange(0, model.T, device=self.device)
            #     ts = ts[None,...].expand(shrinked_batch_size,-1).reshape(-1)

            #     diff_loss, pred_frame = model.compute_loss(last_frame_expanded, ground_truth_expanded, ts, extra_info)
            #     loss = self.diffusion_loss_weight * diff_loss

            else:
                K = self.window_size
            
                if st_index == 0:
                    last_frame = sampled_frames[:,0,:]   # (B,D)
                else:
                    if pred_window is None:
                        last_frame = sampled_frames[:, st_index, :]
                    else:
                        last_frame = pred_window.detach()[:, 0, :]
            
                    teacher_forcing_mask = torch.bernoulli(1.0-torch.ones(batch_size, device=last_frame.device) * sch_samp_prob).bool()
                    last_frame[teacher_forcing_mask] = sampled_frames[teacher_forcing_mask, st_index, :]
            
                # Window ground truth for this step:
                # from frame (st_index+1) up to (st_index+K)
                gt_start = st_index + 1
                gt_end = gt_start + K
                ground_truth = sampled_frames[:, gt_start:gt_end, :]  # (B,K,D) (or shorter near end)
            
                # If near end of sequence, skip
                if ground_truth.shape[1] < K:
                    # continue
                    break
            
                diff_loss_teacher, _ = model.compute_loss(sampled_frames[:, st_index, :], ground_truth, None, extra_info)
                diff_loss_student, pred_window = model.compute_loss(last_frame, ground_truth, None, extra_info)
                # reshape pred_window back to (B, T, Kwin, D) and take ts=0 slice for rollout state
                pred_window = pred_window.reshape(batch_size, model.T, Kwin, -1)[:, 0, :, :]
            
                diff_loss = diff_loss_student + diff_loss_teacher
                loss = self.diffusion_loss_weight * diff_loss
             
            loss.backward()
            self.optimizer.step()
            model.update()

            loss_diff_sum += diff_loss.item()
           
        
        return {"diff_loss": loss_diff_sum}
    

    def train_loop(self, ep, model):
        ep_loss_dict = {}

        num_samples = 0
        self._update_lr_schedule(self.optimizer, ep - 1)
        
        model.train()
        pbar = tqdm(self.train_dataloader, colour='green')
        cur_samples = 1
        for frames in pbar:
            extra_info = None
            frames = frames.to(self.device).float()
            
            self.optimizer.zero_grad()

            if self.sample_schedule[ep]>0:
                loss_dict = self.compute_student_loss(model, frames, self.sample_schedule[ep], extra_info=extra_info)   
            else:
                loss_dict= self.compute_teacher_loss(model, frames, extra_info=extra_info)
            
            
            num_samples += cur_samples
            
            loss = 0
            for key in loss_dict:
                loss += loss_dict[key]
                if key not in ep_loss_dict:
                    ep_loss_dict[key] = loss_dict[key]
                else:
                    ep_loss_dict[key] += loss_dict[key]
                
            
            out_str = ' '.join(['{}:{:.4f}'.format(key,val) for key, val in loss_dict.items()])            
            pbar.set_description('ep:{}, {}'.format(ep, out_str))

        for key in loss_dict:
            ep_loss_dict[key] /= num_samples

        train_info = {
                    "epoch": ep,
                    "sch_smp_rate": self.sample_schedule[ep],
                    **ep_loss_dict
                }
        
        return train_info
