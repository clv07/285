import copy
import numpy as np
import math
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as f
from functools import partial
from copy import deepcopy
import random

import model.model_base as model_base
import model.modules.EMA as EMA 
import model.modules.Embedding as Embedding
import model.modules.Activation as Activation

import dataset.util.geo as geo_util

def enable_mc_dropout(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()



class DropoutModel(model_base.BaseModel):
    NAME = "MCDropout"
    def __init__(self, config, dataset, device):
        super().__init__(config, dataset, device)
       
        #self.estimate_mode = config["diffusion"]["estimate_mode"]   
        #self.loss_type = config["diffusion"]["loss_type"] 
        
        #self.T = config["diffusion"]["T"] 
        #self.sample_mode = config["diffusion"]["sample_mode"]  
        #self.eval_T = config["diffusion"]["eval_T"] if self.sample_mode == 'ddim' else self.T #self.T

        self.frame_dim = dataset.frame_dim
        config['frame_dim'] = self.frame_dim
        self._build_model(config['model_hyperparam'])

        self.use_ema = config["optimizer"].get("EMA", False)
        if self.use_ema:
            print("Using EMA")
            self.ema_step = 0
            self.ema_decay = config['optimizer']['EMA']['ema_decay']
            self.ema_start = config['optimizer']['EMA']['ema_start']
            self.ema_update_rate = config['optimizer']['EMA']['ema_update_rate']
            self.ema_model = deepcopy(self.model)
            self.ema = EMA.EMA(self.ema_decay)
        return

    def forward(self, x):
        return self.model(x)

    def _build_model(self, config):
        frame_size = self.frame_dim
        hidden_size = config['hidden_size']
        layer_num = config['layer_num']
        norm_type = config['norm_type']
        dropout_p = config['dropout_p']
        act_type = config['act_type']
        
        self.model = MLP(frame_size, hidden_size, layer_num, norm_type, dropout_p, act_type)
        self.model.to(self.device)
        return

    def eval_step(self, cur_x, extra_dict=None, align_rpr=False, record_process=False): 
        model = self.ema_model if self.use_ema else self.model
        enable_mc_dropout(model)
        with torch.no_grad():
            next_x = model(cur_x)

        if align_rpr:
            next_x = self.align_frame_with_angle(cur_x, next_x).type(cur_x.dtype)

        return next_x

    def rl_step(self, start_x, action_dict, extra_dict):
        #diffusion = self.ema_diffusion if self.use_ema else self.diffusion 
        #return diffusion.sample_rl_ddpm(start_x, action_dict, extra_dict)
        pass

    
    def eval_seq(self, start_x, extra_dict, num_steps, num_trials, align_rpr=False, record_process=False):
        """
        start_x: (D,) or (B,D)
        returns:
          if not record_process: (B,K,T,D)
          else: (B,K,T,self.T,D)  # matching your original semantics
        """
        if len(start_x.shape) == 1:
            start_x = start_x[None, :]  # (1,D)
    
        B, D = start_x.shape
        K = num_trials
    
        # Expand to (B,K,D) then flatten to (B*K,D)
        x = start_x[:, None, :].expand(B, K, D).reshape(B * K, D)
    
        if record_process:
            output_xs = torch.zeros((B * K, num_steps, self.T, self.frame_dim), device=self.device, dtype=x.dtype)
        else:
            output_xs = torch.zeros((B * K, num_steps, self.frame_dim), device=self.device, dtype=x.dtype)
    
        with torch.inference_mode():
            for j in tqdm(range(num_steps)):
                x = self.eval_step(x, extra_dict, align_rpr, record_process)
    
                output_xs[:, j, ...] = x
    
                if record_process:
                    x = x[..., -1, :]
    
        # Reshape back to (B,K, ...)
        if record_process:
            return output_xs.reshape(B, K, num_steps, self.T, self.frame_dim)
        else:
            return output_xs.reshape(B, K, num_steps, self.frame_dim)

    def eval_step_interactive(self, cur_x, edited_mask, edit_data, extra_dict): 
        #model = self.ema_model if self.use_ema else self.model
        pass
        #if self.sample_mode == 'ddpm':
            #return diffusion.sample_ddpm_interactive(cur_x, edited_mask, edit_data, extra_dict)
        #elif self.sample_mode == 'ddim':
        #    return self.model.sample_ddim_interactive(cur_x, self.eval_T, edited_data, edited_mask, extra_dict)
        #else:
        #    assert(False), "Unsupported agent: {}".format(self.estimate_mode)                

    def eval_seq_interactive(self, start_x, extra_dict, edit_data, edited_mask, num_steps, num_trials):
        """output_xs = torch.zeros((num_trials, num_steps, self.frame_dim)).to(self.device)
        start_x = start_x[None,:].expand(num_trials, -1)
        for j in range(num_steps):
            with torch.no_grad():
                start_x = self.eval_step_interactive(start_x, edit_data[j], edited_mask[j], extra_dict).detach()
            output_xs[:,j,:] = start_x 
        return output_xs"""
        pass



    
    def get_model_params(self):
        params = list(self.model.parameters())
        return params

    def update(self):
        if self.use_ema:
            self.update_ema()

    def update_ema(self):
        self.ema_step += 1
        if self.ema_step % self.ema_update_rate == 0:
            if self.ema_step < self.ema_start:
                self.ema_model.load_state_dict(self.model.state_dict())
            else:
                self.ema.update_model_average(self.ema_model, self.model)


class MLP(nn.Module):
    def __init__(
        self,
        frame_size,
        hidden_size,
        layer_num,
        norm_type,
        dropout_p,
        act_type
    ):
        super().__init__()

        self.input_size = frame_size
        self.dropout = nn.Dropout(dropout_p)
        layers = []
        for _ in range(layer_num): 
            if act_type == 'ReLU':
                non_linear = torch.nn.ReLU() ### v12 is ReLU
            elif act_type == 'SiLU':
                non_linear = Activation.SiLU() 
            linear = nn.Linear(hidden_size + frame_size, hidden_size)
            if norm_type == 'layer_norm':
                norm_layer = nn.LayerNorm(hidden_size)
            elif norm_type == 'group_norm':
                norm_layer = nn.GroupNorm(16, hidden_size)

            layers.append(norm_layer)
            layers.extend([non_linear, linear])
            
        self.net = nn.ModuleList(layers)
        self.fin = nn.Linear(frame_size, hidden_size)
        self.fco = nn.Linear(hidden_size + frame_size, frame_size)
        self.act = Activation.SiLU()
  
    def forward(self, xcur):
        
        y0 = xcur
        
        x = xcur
        x = self.fin(x)
        x = self.dropout(x)

        for i, layer in enumerate(self.net):
            if i % 3 == 2:
                x = self.dropout(x)
                x = torch.cat([x, y0], dim=-1)
                x = layer(x)
            else:
                x = layer(x)

        x = self.dropout(x)
        x = torch.cat([x, y0],dim=-1) 
        x = self.fco(x)
        return x 