import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Parameter as P
from torch.nn import init
from torch.nn.parameter import Parameter
import numpy as np
import lreq as ln
import math
from registry import *

@GENERATORS.register("GeneratorFormant")
class FormantSysth(nn.Module):
   def __init__(self, n_mels=64, k=30):
      super(FormantSysth, self).__init__()
      self.n_mels = n_mels
      self.k = k
      self.timbre = Parameter(torch.Tensor(1,1,n_mels))
    #   self.silient = Parameter(torch.Tensor(1,1,n_mels))
      self.silient = -1
      with torch.no_grad():
         nn.init.constant_(self.timbre,1.0)
        #  nn.init.constant_(self.silient,-1.0)

   def formant_mask(self,freq,bandwith,amplitude):
      # freq, bandwith, amplitude: B*formants*time
      freq_cord = torch.arange(self.n_mels)
      time_cord = torch.arange(freq.shape[2])
      grid_time,grid_freq = torch.meshgrid(time_cord,freq_cord)
      grid_time = grid_time.unsqueeze(dim=0).unsqueeze(dim=-1) #B,time,freq, 1
      grid_freq = grid_freq.unsqueeze(dim=0).unsqueeze(dim=-1) #B,time,freq, 1
      freq = freq.permute([0,2,1]).unsqueeze(dim=-2) #B,time,1, formants
      bandwith = bandwith.permute([0,2,1]).unsqueeze(dim=-2) #B,time,1, formants
      amplitude = amplitude.permute([0,2,1]).unsqueeze(dim=-2) #B,time,1, formants
      masks = amplitude*torch.exp(-(grid_freq-freq)**2/(2*bandwith**2)) #B,time,freqchans, formants
      masks = masks.unsqueeze(dim=1) #B,1,time,freqchans
      return masks

   def mel_scale(self,hz):
      return (torch.log2(hz/440)+31/24)*24*self.n_mels/126
   
   def inverse_mel_scale(self,mel):
      return 440*2**(mel*126/24-31/24)

   def voicing(self,f0):
      #f0: B*1*time
      freq_cord = torch.arange(self.n_mels)
      time_cord = torch.arange(f0.shape[2])
      grid_time,grid_freq = torch.meshgrid(time_cord,freq_cord)
      grid_time = grid_time.unsqueeze(dim=0).unsqueeze(dim=-1) #B,time,freq, 1
      grid_freq = grid_freq.unsqueeze(dim=0).unsqueeze(dim=-1) #B,time,freq, 1
      f0 = f0.permute([0,2,1]).unsqueeze(dim=-2) #B,time,1, 1
      f0 = f0.repeat([1,1,1,self.k]) #B,time,1, self.k
      f0 = f0*(torch.arange(self.k)+1).reshape([1,1,1,self.k])
      bandwith = 24.7*(f0*4.37/1000+1)
      bandwith_lower = torch.clamp(f0-bandwith/2,min=0.001)
      bandwith_upper = f0+bandwith/2
      bandwith = self.mel_scale(bandwith_upper) - self.mel_scale(bandwith_lower)
      f0 = self.mel_scale(f0)
      # hamonics = torch.exp(-(grid_freq-f0)**2/(2*bandwith**2)) #gaussian
      hamonics = (1-((grid_freq-f0)/(3*bandwith/2))**2)*(-torch.sign(torch.abs(grid_freq-f0)/(3*bandwith)-0.5)*0.5+0.5) #welch
      # hamonics = torch.cos(np.pi*torch.abs(grid_freq-f0)/(4*bandwith))**2*(-torch.sign(torch.abs(grid_freq-f0)/(4*bandwith)-0.5)*0.5+0.5) #hanning
      hamonics = (hamonics.sum(dim=-1)*self.timbre).unsqueeze(dim=1) # B,1,T,F
      return hamonics

   def unvoicing(self,f0):
      return torch.ones([f0.shape[0],1,f0.shape[2],self.n_mels])

   def forward(self,components):
      # f0: B*1*T, amplitudes: B*2(voicing,unvoicing)*T, freq_formants,bandwidth_formants,amplitude_formants: B*formants*T
      amplitudes = components['amplitudes'].unsqueeze(dim=-1)
      loudness = components['loudness'].unsqueeze(dim=-1)
      f0_hz = self.inverse_mel_scale(components['f0'])
      self.hamonics = self.voicing(f0_hz)
      self.noise = self.unvoicing(f0_hz)
      freq_formants = components['freq_formants']*self.n_mels
      bandwidth_formants = components['bandwidth_formants']*self.n_mels
      # excitation = amplitudes[:,0:1]*hamonics
      # excitation = loudness*(amplitudes[:,0:1]*hamonics)
      self.excitation = loudness*(amplitudes[:,0:1]*self.hamonics + amplitudes[:,-1:]*self.noise)
      self.mask = self.formant_mask(freq_formants,bandwidth_formants,components['amplitude_formants'])
      self.mask_sum = self.mask.sum(dim=-1)
      speech = self.excitation*self.mask_sum + self.silient*torch.ones(self.mask_sum.shape)
      return speech

@ENCODERS.register("EncoderFormant")
class FormantEncoder(nn.Module):
   def __init__(self, n_mels=64, n_formants=4):
      super(FormantEncoder, self).__init__()
      self.n_mels = n_mels
      self.conv1 = ln.Conv1d(n_mels,64,3,1,1)
      self.norm1 = nn.GroupNorm(32,64)
      self.conv2 = ln.Conv1d(64,128,3,1,1)
      self.norm2 = nn.GroupNorm(32,128)

      self.conv_fundementals = ln.Conv1d(128,128,3,1,1)
      self.norm_fundementals = nn.GroupNorm(32,128)
      self.conv_f0 = ln.Conv1d(128,1,1,1,0)
      self.conv_amplitudes = ln.Conv1d(128,2,1,1,0)
      # self.conv_loudness = ln.Conv1d(128,1,1,1,0)

      self.conv_formants = ln.Conv1d(128,128,3,1,1)
      self.norm_formants = nn.GroupNorm(32,128)
      self.conv_formants_freqs = ln.Conv1d(128,n_formants,1,1,0)
      self.conv_formants_bandwidth = ln.Conv1d(128,n_formants,1,1,0)
      self.conv_formants_amplitude = ln.Conv1d(128,n_formants,1,1,0)

      self.amplifier = Parameter(torch.Tensor(1))
      with torch.no_grad():
         nn.init.constant_(self.amplifier,1.0)
   
   def forward(self,x):
      x = x.squeeze(dim=1).permute(0,2,1) #B * f * T
      loudness = torch.mean(x*0.5+0.5,dim=1,keepdim=True)
      loudness = F.softplus(self.amplifier)*loudness
      x = F.leaky_relu(self.norm1(self.conv1(x)),0.2)
      x_common = F.leaky_relu(self.norm2(self.conv2(x)),0.2)

      # loudness = F.relu(self.conv_loudness(x_common))
      amplitudes = F.softmax(self.conv_amplitudes(x_common),dim=1)

      x_fundementals = F.leaky_relu(self.norm_fundementals(self.conv_fundementals(x_common)),0.2)
      # f0 = F.sigmoid(self.conv_f0(x_fundementals))
      # f0 = F.tanh(self.conv_f0(x_fundementals)) * (16/64)*(self.n_mels/64) # 72hz < f0 < 446 hz
      f0 = F.sigmoid(self.conv_f0(x_fundementals)) * (15/64)*(self.n_mels/64) # 179hz < f0 < 420 hz
      # f0 = F.sigmoid(self.conv_f0(x_fundementals)) * (22/64)*(self.n_mels/64) - (16/64)*(self.n_mels/64)# 72hz < f0 < 253 hz, human voice
      # f0 = F.sigmoid(self.conv_f0(x_fundementals)) * (11/64)*(self.n_mels/64) - (-2/64)*(self.n_mels/64)# 160hz < f0 < 300 hz, female voice
      x_formants = F.leaky_relu(self.norm_formants(self.conv_formants(x_common)),0.2)
      formants_freqs = F.sigmoid(self.conv_formants_freqs(x_formants))
      formants_freqs = torch.cumsum(formants_freqs,dim=1)
      formants_freqs = formants_freqs
      # formants_freqs = formants_freqs + f0
      formants_bandwidth = F.sigmoid(self.conv_formants_bandwidth(x_formants))
      formants_amplitude = F.softmax(self.conv_formants_amplitude(x_formants),dim=1)

      components = { 'f0':f0,
                     'loudness':loudness,
                     'amplitudes':amplitudes,
                     'freq_formants':formants_freqs,
                     'bandwidth_formants':formants_bandwidth,
                     'amplitude_formants':formants_amplitude,
      }
      return components

class FromECoG(nn.Module):
    def __init__(self, outputs,residual=False):
        super().__init__()
        self.residual=residual
        self.from_ecog = ln.Conv3d(1, outputs, [9,1,1], 1, [4,0,0])

    def forward(self, x):
        x = self.from_ecog(x)
        if not self.residual:
            x = F.leaky_relu(x, 0.2)
        return x

class ECoGMappingBlock(nn.Module):
    def __init__(self, inputs, outputs, kernel_size,dilation=1,fused_scale=True,residual=False,resample=[]):
        super(ECoGMappingBlock, self).__init__()
        self.residual = residual
        self.inputs_resample = resample
        self.dim_missmatch = (inputs!=outputs)
        self.resample = resample
        if not self.resample:
            self.resample=1
        self.padding = list(np.array(dilation)*(np.array(kernel_size)-1)//2)
        # self.padding = [dilation[i]*(kernel_size[i]-1)//2 for i in range(len(dilation))]
        if residual:
            self.norm1 = nn.GroupNorm(min(inputs,32),inputs)
        else:
            self.norm1 = nn.GroupNorm(min(outputs,32),outputs)
        self.conv1 = ln.Conv3d(inputs, outputs, kernel_size, self.resample, self.padding, dilation=dilation, bias=False)
        if self.inputs_resample or self.dim_missmatch:
            self.convskip = ln.Conv3d(inputs, outputs, kernel_size, self.resample, self.padding, dilation=dilation, bias=False)
                
        self.conv2 = ln.Conv3d(outputs, outputs, kernel_size, 1, self.padding, dilation=dilation, bias=False)
        self.norm2 = nn.GroupNorm(min(outputs,32),outputs)

    def forward(self,x):
        if self.residual:
            x = F.leaky_relu(self.norm1(x),0.2)
            if self.inputs_resample or self.dim_missmatch:
                # x_skip = F.avg_pool3d(x,self.resample,self.resample)
                x_skip = self.convskip(x)
            else:
                x_skip = x
            x = F.leaky_relu(self.norm2(self.conv1(x)),0.2)
            x = self.conv2(x)
            x = x_skip + x
        else:
            x = F.leaky_relu(self.norm1(self.conv1(x)),0.2)
            x = F.leaky_relu(self.norm2(self.conv2(x)),0.2)
        return x


@ECOG_ENCODER.register("ECoGMappingBottleneck")
class ECoGMapping_Bottleneck(nn.Module):
    def __init__(self,n_mels,n_formants):
        super(ECoGMapping_Bottleneck, self).__init__()
        self.n_formants = n_formants
        self.n_mels = n_mels
        self.from_ecog = FromECoG(16,residual=True)
        self.conv1 = ECoGMappingBlock(16,32,[5,1,1],residual=True,resample = [2,1,1])
        self.conv2 = ECoGMappingBlock(32,64,[3,1,1],residual=True,resample = [2,1,1])
        self.norm_mask = nn.GroupNorm(32,64) 
        self.mask = ln.Conv3d(64,1,[3,1,1],1,[1,0,0])
        self.conv3 = ECoGMappingBlock(64,128,[3,3,3],residual=True,resample = [2,2,2])
        self.conv4 = ECoGMappingBlock(128,256,[3,3,3],residual=True,resample = [2,2,2])
        self.norm = nn.GroupNorm(32,256)
        self.conv5 = ln.Conv1d(256,256,3,1,1)
        self.norm2 = nn.GroupNorm(32,256)
        self.conv6 = ln.ConvTranspose1d(256, 128, 3, 2, 1, transform_kernel=True)
        self.norm3 = nn.GroupNorm(32,128)
        self.conv7 = ln.ConvTranspose1d(128, 64, 3, 2, 1, transform_kernel=True)
        self.norm4 = nn.GroupNorm(32,64)
        self.conv8 = ln.ConvTranspose1d(64, 32, 3, 2, 1, transform_kernel=True)
        self.norm5 = nn.GroupNorm(32,32)
        self.conv9 = ln.ConvTranspose1d(32, 32, 3, 2, 1, transform_kernel=True)
        self.norm6 = nn.GroupNorm(32,32)

        self.conv_fundementals = ln.Conv1d(32,32,3,1,1)
        self.norm_fundementals = nn.GroupNorm(32,32)
        self.conv_f0 = ln.Conv1d(32,1,1,1,0)
        self.conv_amplitudes = ln.Conv1d(32,2,1,1,0)
        self.conv_loudness = ln.Conv1d(32,1,1,1,0)

        self.conv_formants = ln.Conv1d(32,32,3,1,1)
        self.norm_formants = nn.GroupNorm(32,32)
        self.conv_formants_freqs = ln.Conv1d(32,n_formants,1,1,0)
        self.conv_formants_bandwidth = ln.Conv1d(32,n_formants,1,1,0)
        self.conv_formants_amplitude = ln.Conv1d(32,n_formants,1,1,0)

    def forward(self,ecog,mask_prior):
        x_common_all = []
        for d in range(len(ecog)):
            x = ecog[d]
            x = x.reshape([-1,1,x.shape[1],15,15])
            mask_prior_d = mask_prior[d].reshape(-1,1,1,15,15)
            x = self.from_ecog(x)
            x = self.conv1(x)
            x = self.conv2(x)
            mask = torch.sigmoid(self.mask(F.leaky_relu(self.norm_mask(x),0.2)))
            mask = mask[:,:,4:]
            if mask_prior is not None:
                mask = mask*mask_prior_d
            x = x[:,:,4:]
            x = x*mask
            x = self.conv3(x)
            x = self.conv4(x)
            x = x.max(-1)[0].max(-1)[0]
            x = self.conv5(F.leaky_relu(self.norm(x),0.2))
            x = self.conv6(F.leaky_relu(self.norm2(x),0.2))
            x = self.conv7(F.leaky_relu(self.norm3(x),0.2))
            x = self.conv8(F.leaky_relu(self.norm4(x),0.2))
            x = self.conv9(F.leaky_relu(self.norm5(x),0.2))
            x_common = F.leaky_relu(self.norm6(x),0.2)
            x_common_all += [x_common]

        x_common = torch.cat(x_common_all,dim=0)
        loudness = F.relu(self.conv_loudness(x_common))
        amplitudes = F.softmax(self.conv_amplitudes(x_common),dim=1)

        x_fundementals = F.leaky_relu(self.norm_fundementals(self.conv_fundementals(x_common)),0.2)
        # f0 = F.sigmoid(self.conv_f0(x_fundementals))
        # f0 = F.tanh(self.conv_f0(x_fundementals)) * (16/64)*(self.n_mels/64) # 72hz < f0 < 446 hz
        f0 = F.sigmoid(self.conv_f0(x_fundementals)) * (15/64)*(self.n_mels/64) # 179hz < f0 < 420 hz
        # f0 = F.sigmoid(self.conv_f0(x_fundementals)) * (22/64)*(self.n_mels/64) - (16/64)*(self.n_mels/64)# 72hz < f0 < 253 hz, human voice
        # f0 = F.sigmoid(self.conv_f0(x_fundementals)) * (11/64)*(self.n_mels/64) - (-2/64)*(self.n_mels/64)# 160hz < f0 < 300 hz, female voice
        x_formants = F.leaky_relu(self.norm_formants(self.conv_formants(x_common)),0.2)
        formants_freqs = F.sigmoid(self.conv_formants_freqs(x_formants))
        formants_freqs = torch.cumsum(formants_freqs,dim=1)
        formants_freqs = formants_freqs
        # formants_freqs = formants_freqs + f0
        formants_bandwidth = F.sigmoid(self.conv_formants_bandwidth(x_formants))
        formants_amplitude = F.softmax(self.conv_formants_amplitude(x_formants),dim=1)

        components = { 'f0':f0,
                     'loudness':loudness,
                     'amplitudes':amplitudes,
                     'freq_formants':formants_freqs,
                     'bandwidth_formants':formants_bandwidth,
                     'amplitude_formants':formants_amplitude,
      }
        return components