# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist
from torch.distributions import Gamma
from torch.distributions import Beta
#import matplotlib.pyplot as plt
#from torch_utils import misc

import json

from torch.nn.functional import logsigmoid

#----------------------------------------------------------------------------
MIN = torch.finfo(torch.float32).tiny
EPS = torch.finfo(torch.float32).eps

#----------------------------------------------------------------------------
# Beta diffusion sampler.
def sigma_sigma(t,beta_d=26.0,beta_min=0.1):
    t = torch.as_tensor(t)
    #return ((0.5 * beta_d * (t ** 2) + beta_min * t).exp() - 1).sqrt()
    return (0.5 * beta_d * (t ** 2) + beta_min * t).expm1().sqrt()

def alpha_alpha(t,beta_d=26.0,beta_min=0.1):
    t = torch.as_tensor(t)
    return (-0.5 * beta_d * (t ** 2) - beta_min * t).exp()   

def log_alpha_log_alpha(t,beta_d=26.0,beta_min=0.1):
    t = torch.as_tensor(t)
    return -0.5 * beta_d * (t ** 2) - beta_min * t 

def bd_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=None, alpha_min=None, eta=None, sigmoid_start=None, sigmoid_end=None, sigmoid_power=None, start_step=None,Scale=None,Shift=None
):
    
    if num_steps>350:
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)

        t_steps = 1-step_indices / (num_steps - 1)*(1-1e-5)

        #t_steps = 0.996**(step_indices / (num_steps - 1)*500)

        # epsilon_s = 1e-5                
        # t_steps =  (epsilon_s + step_indices/(num_steps-1) * (1.0 - epsilon_s)).flip(dims=(0,))

        t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

        logit_alpha = sigmoid_start + (sigmoid_end-sigmoid_start) * (t_steps**sigmoid_power)
        alpha = logit_alpha.sigmoid()
    
    else:
        step_indices = torch.arange(num_steps+1, dtype=torch.float64, device=latents.device)
        log_pi = (sigmoid_start + (sigmoid_end - sigmoid_start) * (torch.tensor(1, dtype=torch.float64, device=latents.device)**sigmoid_power)).sigmoid().log() / (num_steps)
        #log_pi = torch.tensor(2e-6, device=latents.device).log()/(num_steps)
        alpha = (step_indices*log_pi).exp()
        alpha = torch.flip(alpha, [0])
        logit_alpha = alpha.logit()
        

    alpha_min = alpha_min if alpha_min is not None else alpha[0]

    if 1:
        log_u = log_gamma( (eta * alpha_min * latents).to(torch.float32) ).to(torch.float64)
        log_v = log_gamma( (eta - eta * alpha_min * latents).to(torch.float32) ).to(torch.float64)
        x_next = (log_u - log_v).to(latents.device)
    else:
        x_next = torch.logit( alpha_min*latents.to(torch.float64) ).to(latents.device)

    # if 1:
    #     x_next =x_next.clamp(torch.log( torch.tensor(MIN,device=latents.device)))

    # Main sampling loop.
    ims = []
    im_xhats = []

    for i, (logit_alpha_cur,logit_alpha_next) in enumerate(zip(logit_alpha[:-1], logit_alpha[1:])): # 0, ..., N-1
        x_cur = x_next
        alpha_cur = logit_alpha_cur.sigmoid()
        alpha_next = logit_alpha_next.sigmoid()

        log_alpha_cur = logsigmoid(logit_alpha_cur)

        xmin = Shift
        xmax = Shift + Scale
        xmean = Shift+Scale/2.0
        
        E1 = 1.0/(eta*alpha_cur*Scale)*((eta * alpha_cur * xmax).lgamma() - (eta * alpha_cur * xmin).lgamma())
        E2 = 1.0/(eta*alpha_cur*Scale)*((eta-eta * alpha_cur * xmin).lgamma() - (eta-eta * alpha_cur * xmax).lgamma())
        E_logit_x_t =  E1 - E2


        V1 = 1.0/(eta*alpha_cur*Scale)*((eta * alpha_cur * xmax).digamma() - (eta * alpha_cur * xmin).digamma())
        V2 = 1.0/(eta*alpha_cur*Scale)*((eta-eta * alpha_cur * xmin).digamma() - (eta-eta * alpha_cur * xmax).digamma())

        grids = (torch.arange(0,101,device=latents.device)/100 +0.5/100)*Scale+Shift
        grids = grids[:-1]
        alpha_x = alpha_cur*grids 
        if 1:
            #V3 = (((eta * alpha_cur * xmean).digamma())**2- E1**2).clamp(0)
            #V4 = (((eta-eta * alpha_cur * xmean).digamma())**2- E2**2).clamp(0)
            #V3 = E1**2
            #V4 = E2**2
            grids = (torch.arange(0,101,device=latents.device)/100)*Scale+Shift
            alpha_x = alpha_cur*grids 
            V3 = ((eta * alpha_x).digamma())**2
            #print(V3.shape)
            V3[0] = (V3[0]+V3[-1])/2
            V3 = V3[:-1]
            V3 = (V3.mean()- E1**2).clamp(0)   
            V4 = ((eta - eta*alpha_x).digamma())**2
            V4[0] = (V4[0]+V4[-1])/2
            V4 = V4[:-1]
            V4 = (V4.mean()- E2**2).clamp(0)
            
            #V3 = (( ((eta * alpha_x).digamma())**2).mean()- E1**2).clamp(0)           
            #V4 = (( ((eta - eta*alpha_x).digamma())**2).mean()- E2**2).clamp(0)
        else:
            V3 = (( ((eta * alpha_x).digamma())**2).mean()- E1**2).clamp(0)           
            V4 = (( ((eta - eta*alpha_x).digamma())**2).mean()- E2**2).clamp(0)


        std_logit_x_t = (V1+V2+V3+V4).sqrt()

        #std_logit_x_t = std_logit_x_t.clamp(EPS)

        logit_x_t = x_cur
        if class_labels is None:
            logit_x0_hat = net((logit_x_t-E_logit_x_t)/std_logit_x_t, logit_alpha_cur).to(torch.float64)
        else:    
            logit_x0_hat = net((logit_x_t-E_logit_x_t)/std_logit_x_t, logit_alpha_cur, class_labels).to(torch.float64)


        x0_hat = torch.sigmoid(logit_x0_hat)* Scale + Shift

        alpha_reverse = (eta*alpha_next-eta*alpha_cur)*x0_hat
        beta_reverse = eta-eta*alpha_next*x0_hat
        log_u = log_gamma(alpha_reverse.to(torch.float32)).to(torch.float64)
        log_v = log_gamma(beta_reverse.to(torch.float32)).to(torch.float64)
        concatenated = torch.cat((x_cur.unsqueeze(-1), (log_u-log_v).unsqueeze(-1), (x_cur+log_u-log_v).unsqueeze(-1)), dim=4)
        x_next = torch.logsumexp(concatenated, dim=4)    

    if 1: # step<num_steps/2:
        out = (x0_hat- Shift) / Scale
        out1 = ((torch.sigmoid(x_next)/alpha_next- Shift) / Scale) #.clamp(0,1)
        # else:
        #     out = 0.8*out+0.2*temp_out 
        #     #out1 = 0.8*out1+0.2*((torch.sigmoid(x_cur)/alpha[step]- Shift) / Scale) #.clamp(0,1)
        #     out1 = 0.8*out1+0.2*((torch.sigmoid(x_next)/alpha[step-1]- Shift) / Scale) #.clamp(0,1)
        #     #out1 = 0.8*out1+0.2*(torch.sigmoid(x_cur)/alpha[step]) #.clamp(0,1)    
    return (out-0.5)/0.5, (out1-0.5)/0.5    


def log_gamma(alpha):
    return torch.log(torch._standard_gamma(alpha))


#----------------------------------------------------------------------------
# EDM sampler (Algorithm 2).

def edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

#----------------------------------------------------------------------------
# Generalized ablation sampler, representing the superset of all sampling
# methods discussed in the paper.

def ablation_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=None, sigma_max=None, rho=7,
    solver='heun', discretization='edm', schedule='linear', scaling='none',
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    assert solver in ['euler', 'heun']
    assert discretization in ['vp', 've', 'iddpm', 'edm']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.9, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == 'vp':
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == 've':
        orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == 'iddpm':
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    else:
        assert discretization == 'edm'
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    # Define noise level schedule.
    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == 'none'
        s = lambda t: 1
        s_deriv = lambda t: 0

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
        t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

        # Euler step.
        h = t_next - t_hat
        denoised = net(x_hat / s(t_hat), sigma(t_hat), class_labels).to(torch.float64)
        d_cur = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
        x_prime = x_hat + alpha * h * d_cur
        t_prime = t_hat + alpha * h

        # Apply 2nd order correction.
        if solver == 'euler' or i == num_steps - 1:
            x_next = x_hat + h * d_cur
        else:
            assert solver == 'heun'
            denoised = net(x_prime / s(t_prime), sigma(t_prime), class_labels).to(torch.float64)
            d_prime = (sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next = x_hat + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)

    return x_next

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)

@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=200, show_default=True)
# added input for beta diffusion




# @click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
# @click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
# @click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
# @click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
# @click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
# @click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
# @click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)

# @click.option('--solver',                  help='Ablate ODE solver', metavar='euler|heun',                          type=click.Choice(['euler', 'heun']))
# @click.option('--disc', 'discretization',  help='Ablate time step discretization {t_i}', metavar='vp|ve|iddpm|edm', type=click.Choice(['vp', 've', 'iddpm', 'edm']))
# @click.option('--schedule',                help='Ablate noise schedule sigma(t)', metavar='vp|ve|linear',           type=click.Choice(['vp', 've', 'linear']))
# @click.option('--scaling',                 help='Ablate signal scaling s(t)', metavar='vp|none',                    type=click.Choice(['vp', 'none']))

#def main(network_pkl, outdir, subdirs, seeds, class_idx, max_batch_size, device=torch.device('cuda'), **sampler_kwargs):
def main(network_pkl, outdir, subdirs, seeds, class_idx, max_batch_size, device=torch.device('cuda'), **sampler_kwargs):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Generate 64 images and save them as out/*.png
    python generate.py --outdir=out --seeds=0-63 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

    \b
    # Generate 1024 images using 2 GPUs
    torchrun --standalone --nproc_per_node=2 generate.py --outdir=out --seeds=0-999 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
    """
    dist.init()
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Loop over batches.
    dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
    
    
    
    #print(network_pkl)


    directory_path = '/'.join(network_pkl.split('/')[:-1])

            # New file name to read
    jason_file_name = "training_options.json"

    # Construct the new file path
    json_file_path = directory_path + '/' + jason_file_name

    # Load the JSON file
    with open(json_file_path, "r") as json_file:
        json_data = json.load(json_file)
    #print(json_data)
    # Access the value of 'eta'

    eta = json_data['loss_kwargs']['eta']
    sampler_kwargs['eta'] = eta
    # beta_d = json_data['loss_kwargs']['beta_d']
    # sampler_kwargs['beta_d']= beta_d
    # beta_min = json_data['loss_kwargs']['beta_min']
    # sampler_kwargs['beta_min']=beta_min
    Scale = json_data['loss_kwargs']['Scale']
    sampler_kwargs['Scale']=Scale
    Shift = json_data['loss_kwargs']['Shift']
    sampler_kwargs['Shift']=Shift
    lossType = json_data['loss_kwargs']['lossType']
    
    
    sigmoid_start = json_data['loss_kwargs']['sigmoid_start']
    sampler_kwargs['sigmoid_start']= sigmoid_start
    sigmoid_end = json_data['loss_kwargs']['sigmoid_end']
    sampler_kwargs['sigmoid_end']= sigmoid_end
    sigmoid_power = json_data['loss_kwargs']['sigmoid_power']
    sampler_kwargs['sigmoid_power']= sigmoid_power
    
    
    # Print the value of 'eta'
    #print(eta,beta_d,beta_min,Scale,Shift,lossType,sampler_kwargs['num_steps'])
    print(eta,sigmoid_start,sigmoid_end,sigmoid_power,Scale,Shift,lossType,sampler_kwargs['num_steps'])
    
    saveImage = True
    
    
    from torchvision.datasets import CIFAR10
    dataset = CIFAR10(root="~/datasets", download=True)
    data = dataset.data / 255.0
    datamean= torch.tensor(data.mean(0)).permute(2, 0, 1)
    
    
    
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        if 0:
            latents = torch.ones([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)

            latents = Beta(latents,1.0).sample()* Scale + Shift
        else:
        #latents = latents*0.5*Scale+Shift
            latents = (datamean*Scale+Shift).expand(batch_size, net.img_channels, net.img_resolution, net.img_resolution).to(device)
        
        #latents = latents *1 #0.5
        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1

        # Generate images.
        sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
        have_ablation_kwargs = any(x in sampler_kwargs for x in ['solver', 'discretization', 'schedule', 'scaling'])
        sampler_fn = ablation_sampler if have_ablation_kwargs else bd_sampler
        #images,images1 = sampler_fn(net, latents, class_labels, randn_like=rnd.randn_like)
        ##
        
        # Path to the JSON file

        images,images1 = sampler_fn(net, latents, class_labels, randn_like=rnd.randn_like, **sampler_kwargs)

        # Save images.
        images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        
        for seed, image_np in zip(batch_seeds, images_np):
            image_dir = os.path.join(outdir, f'{seed-seed%1000:06d}') if subdirs else outdir
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f'{seed:06d}.png')
            if image_np.shape[2] == 1:
                PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
            else:
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)
        if saveImage:
            n = len(images_np)
            m = int(np.ceil(np.sqrt(n)))
            w = h = net.img_resolution
            grid = np.zeros((m*h, m*w, 3), dtype=np.uint8)
            for i, i_np in enumerate(images_np):
                x, y = i%m, i//m
                grid[y*h:(y+1)*h, x*w:(x+1)*w, :] = i_np

            image_grid = PIL.Image.fromarray(grid, 'RGB')
            #display(image_grid)  
            #plt.imshow(image_grid)
            #plt.show()
            random_number = np.random.rand()
            # Saving the image with a random filename
            image_grid.save(f"{network_pkl[0:-4]}_{random_number:.6f}.png")
        
        images_np = (images1 * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        outdir1 = outdir+'_1'
        for seed, image_np in zip(batch_seeds, images_np):
            image_dir = os.path.join(outdir1, f'{seed-seed%1000:06d}') if subdirs else outdir1
            os.makedirs(image_dir, exist_ok=True)
            image_path = os.path.join(image_dir, f'{seed:06d}.png')
            if image_np.shape[2] == 1:
                PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
            else:
                PIL.Image.fromarray(image_np, 'RGB').save(image_path)
        if saveImage:
            n = len(images_np)
            m = int(np.ceil(np.sqrt(n)))
            w = h = net.img_resolution
            grid = np.zeros((m*h, m*w, 3), dtype=np.uint8)
            for i, i_np in enumerate(images_np):
                x, y = i%m, i//m
                grid[y*h:(y+1)*h, x*w:(x+1)*w, :] = i_np

            image_grid = PIL.Image.fromarray(grid, 'RGB')
            #display(image_grid)  
            #plt.imshow(image_grid)
            #plt.show()
            random_number = np.random.rand()
            # Saving the image with a random filename
            image_grid.save(f"{network_pkl[0:-4]}_{random_number:.6f}_1.png")
        saveImage = False
                
    
#     imagelist = [os.path.join(image_path,im) for im in os.listdir(image_path) if im.endswith(".png")]
#     imagelist = [np.array(Image.open(im)) for im in imagelist][0:64]
#     im = np.stack(imagelist).reshape(len(imagelist)//8,8,32,32,3).transpose(0,2,1,3,4).reshape(-1,32*8,3)
#     #im = np.stack(imagelist).reshape(len(imagelist)//5,5,32,32,3).transpose(0,2,1,3,4).reshape(-1,32*5,3)
#     #plt.figure(figsize=(10,10))
#     #plt.imshow(im)
#     #plt.savefig('plot10.png')

#     im = imagelist
#     n = len(im)
#     m = int(np.ceil(np.sqrt(n)))
#     w = h = net.img_resolution
#     grid = np.zeros((m*h, m*w, 3), dtype=np.uint8)
#     for i, i_np in enumerate(im):
#         x, y = i%m, i//m
#         grid[y*h:(y+1)*h, x*w:(x+1)*w, :] = i_np
#     image_grid = PIL.Image.fromarray(grid, 'RGB')
#     display(image_grid)  
#     image_grid.save("temp.png")
    # Done.
    torch.distributed.barrier()
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
