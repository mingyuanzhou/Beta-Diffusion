# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""





import torch

EPS=torch.finfo(torch.float32).eps
MIN=torch.finfo(torch.float32).tiny

from torch_utils import persistence

from torch.nn.functional import logsigmoid

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

@persistence.persistent_class
class BetaDiffLoss:
    def __init__(self, eta=None, sigmoid_start=None, sigmoid_end=None, sigmoid_power=None, Scale=None, Shift=None, T=200, epsilon_t=1e-5,lossType='KLUB'):
    
        self.eta = eta
        self.sigmoid_start = sigmoid_start
        self.sigmoid_end = sigmoid_end
        self.sigmoid_power = sigmoid_power
        self.Scale = Scale
        self.Shift = Shift
        self.T = T
        self.lossType = lossType
        self.epsilon_t = epsilon_t
        self.min = torch.finfo(torch.float32).tiny
        self.eps = torch.finfo(torch.float32).eps

    def __call__(self, net, images, labels=None, augment_pipe=None):
        if 1:
            rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
            rnd_position = 1 + rnd_uniform * (self.epsilon_t - 1)
            #self.epsilon_t + rnd_uniform * (1.0-self.epsilon_t)
            logit_alpha = self.sigmoid_start + (self.sigmoid_end-self.sigmoid_start) * (rnd_position**self.sigmoid_power)
            #rnd_position_previous = (rnd_position - 1/self.T).clamp(min=0)
            rnd_position_previous = rnd_position*0.95
            logit_alpha_previous = self.sigmoid_start + (self.sigmoid_end-self.sigmoid_start) * (rnd_position_previous**self.sigmoid_power)

            alpha = logit_alpha.sigmoid()
            alpha_previous = logit_alpha_previous.sigmoid()

            delta  = (logit_alpha_previous.to(torch.float64).sigmoid()-logit_alpha.to(torch.float64).sigmoid()).to(torch.float32)
        else:
            rnd_position = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
            rnd_position_previous = rnd_position*0.95
            #alpha = (1e-6)**rnd_position
            
            alpha = (torch.tensor(2e-6,device=images.device).log().to(torch.float64)*rnd_position).exp()
            alpha_previous = (torch.tensor(1e-6,device=images.device).log().to(torch.float64)*rnd_position_previous).exp()
            
            delta = (alpha_previous-alpha).to(torch.float32)
            
            
            if 0: 
                logit_alpha_previous = self.sigmoid_start + (self.sigmoid_end-self.sigmoid_start) * (rnd_position**self.sigmoid_power)
                alpha_previous = logit_alpha_previous.sigmoid()
            
                alpha = alpha_previous*0.95
                delta = alpha_previous*0.05
            
            logit_alpha = alpha.logit().to(torch.float32)
            alpha = alpha.to(torch.float32)
            
        eta = torch.ones([images.shape[0], 1, 1, 1], device=images.device) * self.eta

        # Prepare x0, xt
        x0, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        
#        x0 = x0 + (2*torch.rand_like(x0)-1)/510/2
#        x0 = 2*x0.clamp(min=0, max=1) - x0

        x0 = ((x0+1.0)/2.0).clamp(0,1) * self.Scale + self.Shift
        #x0 = (((x0+1.0)/2.0) * self.Scale + self.Shift).clamp(0,1)
    
 
        log_u = self.log_gamma( (self.eta * alpha * x0).to(torch.float32))
        log_v = self.log_gamma( (self.eta - self.eta * alpha * x0).to(torch.float32))

        logit_x_t = (log_u - log_v).to(images.device) 
    
 
        #log_alpha = logsigmoid(logit_alpha)
        xmin = self.Shift
        xmax = self.Shift + self.Scale
        xmean = self.Shift+self.Scale/2.0
        E1 = 1.0/(self.eta*alpha*self.Scale)*((self.eta * alpha * xmax).lgamma() - (self.eta * alpha * xmin).lgamma())
        E2 = 1.0/(self.eta*alpha*self.Scale)*((self.eta-self.eta * alpha * xmin).lgamma() - (self.eta-self.eta * alpha * xmax).lgamma())
        E_logit_x_t =  E1 - E2
        #print(alpha.shape)
        #print(E1.shape)
        V1 = 1.0/(self.eta*alpha*self.Scale)*((self.eta * alpha * xmax).digamma() - (self.eta * alpha * xmin).digamma())
        V2 = 1.0/(self.eta*alpha*self.Scale)*((self.eta-self.eta * alpha * xmin).digamma() - (self.eta-self.eta * alpha * xmax).digamma())
        if 1:
            #V3 = (((self.eta * alpha * xmean).digamma())**2- E1**2).clamp(0)
            #V4 = (((self.eta-self.eta * alpha * xmean).digamma())**2- E2**2).clamp(0)
            #V3 = E1**2
            #V4 = E2**2
            grids = (torch.arange(0,101,device=images.device)/100)*self.Scale+self.Shift
            #grids = (torch.arange(0,1001,device=images.device)/1000 +0.5/1000)*self.Scale+self.Shift
            #grids = grids[:-1]
            alpha_x = alpha[:,:,0,0]*grids.unsqueeze(0)
            #print(alpha_x.shape)
            
            V3 =  ((self.eta * alpha_x).digamma())**2
            V3[:,0] = (V3[:,0]+V3[:,-1])/2
            V3 = V3[:,:-1]
            V3 = (V3.mean(dim=1).unsqueeze(1).unsqueeze(2).unsqueeze(3)- E1**2).clamp(0)  
            
            V4 = ((self.eta - self.eta*alpha_x).digamma())**2
            V4[:,0] = (V4[:,0]+V4[:,-1])/2
            V4 = V4[:,:-1]
            V4 = (V4.mean(dim=1).unsqueeze(1).unsqueeze(2).unsqueeze(3)- E2**2).clamp(0)
            
            #V3 = (( ((self.eta * alpha_x).digamma())**2).mean(dim=1).unsqueeze(1).unsqueeze(2).unsqueeze(3)- E1**2).clamp(0)           
            #V4 = (( ((self.eta - self.eta*alpha_x).digamma())**2).mean(dim=1).unsqueeze(1).unsqueeze(2).unsqueeze(3)- E2**2).clamp(0)

        #print(V1.shape)
        else:
            grids = (torch.arange(0,101,device=images.device)/100 +0.5/100)*self.Scale+self.Shift
            #grids = (torch.arange(0,1001,device=images.device)/1000 +0.5/1000)*self.Scale+self.Shift
            grids = grids[:-1]
            alpha_x = alpha[:,:,0,0]*grids.unsqueeze(0)
            #print(alpha_x.shape)
            V3 = (( ((self.eta * alpha_x).digamma())**2).mean(dim=1).unsqueeze(1).unsqueeze(2).unsqueeze(3)- E1**2).clamp(0)           
            V4 = (( ((self.eta - self.eta*alpha_x).digamma())**2).mean(dim=1).unsqueeze(1).unsqueeze(2).unsqueeze(3)- E2**2).clamp(0)

        #print(V3.shape)
        #print(V4.shape)
        #print(logit_x_t.shape)
        #V3 = (( ((eta * alpha[:,:,0,0] * (torch.arrange(0,100)/99*self.Scale+self.Shift).unsqeeuze(0) ).digamma())**2).mean(dim=1)- E1**2).clamp(0)
        #V4 = (( ((eta - eta*alpha[:,:,0,0] * (torch.arrange(0,100)/99*self.Scale+self.Shift).unsqeeuze(0) ).digamma())**2).mean(dim=1)- E2**2).clamp(0)


        std_logit_x_t = (V1+V2+V3+V4).sqrt()

        #std_logit_x_t = std_logit_x_t.clamp(EPS)


        # Var_logit_x_t = 1.0/(eta*alpha*self.Scale)*((eta * alpha * xmax).digamma() - (eta * alpha * xmin).digamma() + (eta-eta * alpha * xmin).digamma() - (eta-eta * alpha * xmax).digamma()) + \
        #                  ((eta * alpha * xmean).digamma())**2- E1**2 + \
        #                  ((eta-eta * alpha * xmean).digamma())**2- E2**2
        # #print(Var_logit_x_t)
        # std_logit_x_t = Var_logit_x_t.sqrt()
        #logit_x0_hat = net((logit_x_t-E_logit_x_t)/std_logit_x_t, log_alpha)
        #logit_x0_hat = net((logit_x_t-E_logit_x_t)/std_logit_x_t, logit_alpha)
        
        # if labels is None:
        #     logit_x0_hat = net((logit_x_t-E_logit_x_t)/std_logit_x_t, logit_alpha)
        # else:
        #     logit_x0_hat = net((logit_x_t-E_logit_x_t)/std_logit_x_t, logit_alpha,labels, augment_labels=augment_labels)
        
        logit_x0_hat = net((logit_x_t-E_logit_x_t)/std_logit_x_t, logit_alpha,labels, augment_labels=augment_labels)
                    
        x0_hat = torch.sigmoid(logit_x0_hat)* self.Scale + self.Shift
    
        
        loss = self.compute_loss(x0, x0_hat, alpha, alpha_previous, eta, delta,logit_x_t)
        
        return loss


    def compute_loss(self, x0, x0_hat, alpha, alpha_previous, eta, delta,logit_x_t):
        alpha_p = eta*delta*x0 
        beta_p = eta-eta*alpha_previous*x0
        alpha_q = eta*delta*x0_hat
        beta_q  = eta-eta*alpha_previous*x0_hat 

        _alpha_p = eta*alpha*x0 
        _beta_p  = eta-eta*alpha*x0
        _alpha_q = eta*alpha*x0_hat
        _beta_q  = eta-eta*alpha*x0_hat 

        KLUB_conditional = (self.KL_gamma(alpha_q,alpha_p).clamp(0)\
                                + self.KL_gamma(beta_q,beta_p).clamp(0)\
                                - self.KL_gamma(alpha_q+beta_q,alpha_p+beta_p).clamp(0)).clamp(0)
        KLUB_marginal = (self.KL_gamma(_alpha_q,_alpha_p).clamp(0)\
                            + self.KL_gamma(_beta_q,_beta_p).clamp(0)\
                            - self.KL_gamma(_alpha_q+_beta_q,_alpha_p+_beta_p).clamp(0)).clamp(0)

        
        KLUB_conditional_AS = (self.KL_gamma(alpha_p,alpha_q).clamp(0)\
                                    + self.KL_gamma(beta_p,beta_q).clamp(0)\
                                    - self.KL_gamma(alpha_p+beta_p,alpha_q+beta_q).clamp(0)).clamp(0)
        KLUB_marginal_AS = (self.KL_gamma(_alpha_p,_alpha_q).clamp(0)\
                                + self.KL_gamma(_beta_p,_beta_q).clamp(0)\
                                - self.KL_gamma(_alpha_p+_beta_p,_alpha_q+_beta_q).clamp(0)).clamp(0)
        #loss = KLUB_marginal
        loss_dict = {
            'KLUB': (.99 * KLUB_conditional + .01 * KLUB_marginal),
            'KLUB_marginal': KLUB_marginal,
            'KLUB_conditional': KLUB_conditional,
            'KLUB_AS': (0.99 * KLUB_conditional_AS + 0.01 * KLUB_marginal_AS),
            'ELBO': 0.99 * KLUB_conditional_AS + 0.01 * KLUB_marginal_AS,
            'KLUB_marginal_AS': KLUB_marginal_AS,
            'SNR_Weighted_ELBO': KLUB_marginal_AS,
            'KLUB_conditional_AS': KLUB_conditional_AS,
            'L2': torch.square(x0 - x0_hat),
            'L1': torch.abs(x0 - x0_hat),
            # Add other loss types here
        }

        if self.lossType not in loss_dict:
            raise NotImplementedError("Loss type not implemented")
        loss = loss_dict[self.lossType]
        
        return loss_dict[self.lossType]
        #return loss
    
    #Define Beta-diffusion functions
    def log_gamma(self, alpha):
        #return torch.log(torch._standard_gamma(alpha).clamp(MIN))
        return torch.log(torch._standard_gamma(alpha.to(torch.float32)).clamp(MIN))
        #return torch.log(torch._standard_gamma(alpha.to(torch.float64))).to(torch.float32)


    def KL_gamma(self, alpha_p, alpha_q, beta_p=None, beta_q=None):
        """
        Calculates the KL divergence between two Gamma distributions.
        alpha_p: the shape of the first Gamma distribution Gamma(alpha_p,beta_p).
        alpha_q: the shape of the second Gamma distribution Gamma(alpha_q,beta_q).
        beta_p (optional): the rate (inverse scale) of the first Gamma distribution Gamma(alpha_p,beta_p).
        beta_q (optional): the rate (inverse scale) of the second Gamma distribution Gamma(alpha_q,beta_q).
        """    
        KL = (alpha_p-alpha_q)*torch.digamma(alpha_p)-torch.lgamma(alpha_p)+torch.lgamma(alpha_q)
        if beta_p is not None and beta_q is not None:
            KL = KL + alpha_q*(torch.log(beta_p)-torch.log(beta_q))+alpha_p*(beta_q/beta_p-1.0)  
        return KL

    def KL_beta(self, alpha_p,beta_p,alpha_q,beta_q):
        """
        Calculates the KL divergence between two Beta distributions
        KL(Beta(alpha_p,beta_p) || Beta(alpha_q,beta_q))
        """
        KL =KL_gamma(alpha_p,alpha_q)+KL_gamma(beta_p,beta_q)-KL_gamma(alpha_p+beta_p,alpha_q+beta_q)
        return KL