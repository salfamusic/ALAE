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
from transformer_models.position_encoding import build_position_encoding
from transformer_models.transformer import Transformer as TransformerTS
from transformer_models.transformer_nonlocal import Transformer as TransformerNL

def mel_scale(n_mels,hz):
    #take absolute hz, return abs mel
    return (torch.log2(hz/440)+31/24)*24*n_mels/126

def inverse_mel_scale(mel):
    #take normalized mel, return absolute hz
    return 440*2**(mel*126/24-31/24)

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
      masks = amplitude*torch.exp(-0.693*(grid_freq-freq)**2/(2*(bandwith+0.001)**2)) #B,time,freqchans, formants
      masks = masks.unsqueeze(dim=1) #B,1,time,freqchans
      return masks

   def voicing(self,f0):
      #f0: B*1*time, hz
      freq_cord = torch.arange(self.n_mels)
      time_cord = torch.arange(f0.shape[2])
      grid_time,grid_freq = torch.meshgrid(time_cord,freq_cord)
      grid_time = grid_time.unsqueeze(dim=0).unsqueeze(dim=-1) #B,time,freq, 1
      grid_freq = grid_freq.unsqueeze(dim=0).unsqueeze(dim=-1) #B,time,freq, 1
      f0 = f0.permute([0,2,1]).unsqueeze(dim=-2) #B,time,1, 1
      f0 = f0.repeat([1,1,1,self.k]) #B,time,1, self.k
      f0 = f0*(torch.arange(self.k)+1).reshape([1,1,1,self.k])
      bandwith = 24.7*(f0*4.37/1000+1)
      bandwith_lower = torch.clamp(f0-bandwith/2,min=1)
      bandwith_upper = f0+bandwith/2
      bandwith = mel_scale(self.n_mels,bandwith_upper) - mel_scale(self.n_mels,bandwith_lower)
      f0 = mel_scale(self.n_mels,f0)
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
      f0_hz = inverse_mel_scale(components['f0'])
      self.hamonics = self.voicing(f0_hz)
      self.noise = self.unvoicing(f0_hz)
      freq_formants = components['freq_formants']*self.n_mels
    #   ### ratio on mel
    #   freq_formants_hz = inverse_mel_scale(freq_formants/self.n_mels)
    #   bandwidth_formants_hz = self.formant_bandwitdh_ratio*freq_formants_hz
    #   bandwith_lower = torch.clamp(freq_formants_hz-bandwidth_formants_hz/2,min=0.001)
    #   bandwith_upper = freq_formants_hz+bandwidth_formants_hz/2
    #   bandwidth_formants = mel_scale(self.n_mels,bandwith_upper) - mel_scale(self.n_mels,bandwith_lower)
    #   ###
    #   ### ratio on hz
    #   bandwidth_formants = self.formant_bandwitdh_ratio*freq_formants 
    #   #####
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
      self.n_formants = n_formants

      self.formant_freq_limits_diff = torch.tensor([950.,2450.,2100.]).reshape([1,3,1]) #freq difference
      self.formant_freq_limits_diff_low = torch.tensor([300.,300.,0.]).reshape([1,3,1]) #freq difference
      self.formant_freq_limits_abs = torch.tensor([950.,2800.,3400.]).reshape([1,3,1]) #freq difference
      self.formant_freq_limits_abs_low = torch.tensor([300.,600.,2700.]).reshape([1,3,1]) #freq difference

      self.formant_bandwitdh_ratio = Parameter(torch.Tensor(1))
      self.formant_bandwitdh_slop = Parameter(torch.Tensor(1))
      with torch.no_grad():
         nn.init.constant_(self.formant_bandwitdh_ratio,0)
         nn.init.constant_(self.formant_bandwitdh_slop,0)

    #   self.formant_freq_limits = torch.cumsum(self.formant_freq_limits_diff,dim=0)
    #   self.formant_freq_limits_mel = torch.cat([torch.tensor([0.]),mel_scale(n_mels,self.formant_freq_limits)/n_mels])
    #   self.formant_freq_limits_mel_diff = torch.reshape(self.formant_freq_limits_mel[1:]-self.formant_freq_limits_mel[:-1],[1,3,1])

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
      # f0 in mel:
      # f0 = F.sigmoid(self.conv_f0(x_fundementals))
      # f0 = F.tanh(self.conv_f0(x_fundementals)) * (16/64)*(self.n_mels/64) # 72hz < f0 < 446 hz
    #   f0 = F.sigmoid(self.conv_f0(x_fundementals)) * (15/64)*(self.n_mels/64) # 179hz < f0 < 420 hz
      # f0 = F.sigmoid(self.conv_f0(x_fundementals)) * (22/64)*(self.n_mels/64) - (16/64)*(self.n_mels/64)# 72hz < f0 < 253 hz, human voice
      # f0 = F.sigmoid(self.conv_f0(x_fundementals)) * (11/64)*(self.n_mels/64) - (-2/64)*(self.n_mels/64)# 160hz < f0 < 300 hz, female voice

      # f0 in hz:
      f0 = F.sigmoid(self.conv_f0(x_fundementals)) * 240 + 180 # 180hz < f0 < 420 hz
      f0 = torch.clamp(mel_scale(self.n_mels,f0)/self.n_mels,min=0.0001)

      x_formants = F.leaky_relu(self.norm_formants(self.conv_formants(x_common)),0.2)
      formants_freqs = F.sigmoid(self.conv_formants_freqs(x_formants))
    # #  relative freq:
    #   formants_freqs_hz = formants_freqs*(self.formant_freq_limits_diff[:,:self.n_formants]-self.formant_freq_limits_diff_low[:,:self.n_formants])+self.formant_freq_limits_diff_low[:,:self.n_formants]
    # #   formants_freqs_hz = formants_freqs*6839
    #   formants_freqs_hz = torch.cumsum(formants_freqs_hz,dim=1)

      # abs freq:
      formants_freqs_hz = formants_freqs*(self.formant_freq_limits_abs[:,:self.n_formants]-self.formant_freq_limits_abs_low[:,:self.n_formants])+self.formant_freq_limits_abs_low[:,:self.n_formants]
    #   formants_freqs_hz = formants_freqs*6839
      formants_freqs = torch.clamp(mel_scale(self.n_mels,formants_freqs_hz)/self.n_mels,min=0)
      
      # formants_freqs = formants_freqs + f0
      # formants_bandwidth_hz = F.sigmoid(self.conv_formants_bandwidth(x_formants)) *6839
      # formants_bandwidth_hz = F.sigmoid(self.conv_formants_bandwidth(x_formants)) * 150
      formants_bandwidth_hz = F.sigmoid(self.conv_formants_bandwidth(x_formants)) * 3*F.sigmoid(self.formant_bandwitdh_ratio)*(0.075*torch.relu(formants_freqs_hz-1000)+100)
      # formants_bandwidth_hz = F.sigmoid(self.conv_formants_bandwidth(x_formants)) * 3*F.sigmoid(self.formant_bandwitdh_ratio)*(0.075*torch.relu(formants_freqs_hz-1000)+50)
      # formants_bandwidth_hz = F.sigmoid(self.conv_formants_bandwidth(x_formants)) * (0.075*3*F.sigmoid(self.formant_bandwitdh_slop)*torch.relu(formants_freqs_hz-1000)+(2*F.sigmoid(self.formant_bandwitdh_ratio)+1)*50)
      # formants_bandwidth_hz = F.sigmoid(self.conv_formants_bandwidth(x_formants)) * 3*F.sigmoid(self.formant_bandwitdh_ratio)*(0.075*F.sigmoid(self.formant_bandwitdh_slop)*torch.relu(formants_freqs_hz-1000)+50)
      formants_bandwidth_upper = formants_freqs_hz+formants_bandwidth_hz/2
      formants_bandwidth_lower = torch.clamp(formants_freqs_hz-formants_bandwidth_hz/2,min=1)
      formants_bandwidth = (mel_scale(self.n_mels,formants_bandwidth_upper) - mel_scale(self.n_mels,formants_bandwidth_lower))/self.n_mels
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
    def __init__(self, outputs,residual=False,shape='3D'):
        super().__init__()
        self.residual=residual
        if shape =='3D':
            self.from_ecog = ln.Conv3d(1, outputs, [9,1,1], 1, [4,0,0])
        else:
            self.from_ecog = ln.Conv2d(1, outputs, [9,1], 1, [4,0])

    def forward(self, x):
        x = self.from_ecog(x)
        if not self.residual:
            x = F.leaky_relu(x, 0.2)
        return x

class ECoGMappingBlock(nn.Module):
    def __init__(self, inputs, outputs, kernel_size,dilation=1,fused_scale=True,residual=False,resample=[],pool=None,shape='3D'):
        super(ECoGMappingBlock, self).__init__()
        self.residual = residual
        self.pool = pool
        self.inputs_resample = resample
        self.dim_missmatch = (inputs!=outputs)
        self.resample = resample
        if not self.resample:
            self.resample=1
        self.padding = list(np.array(dilation)*(np.array(kernel_size)-1)//2)
        if shape=='2D':
            conv=ln.Conv2d
            maxpool = nn.MaxPool2d
            avgpool = nn.AvgPool2d
        if shape=='3D':
            conv=ln.Conv3d
            maxpool = nn.MaxPool3d
            avgpool = nn.AvgPool3d
        # self.padding = [dilation[i]*(kernel_size[i]-1)//2 for i in range(len(dilation))]
        if residual:
            self.norm1 = nn.GroupNorm(min(inputs,32),inputs)
        else:
            self.norm1 = nn.GroupNorm(min(outputs,32),outputs)
        if pool is None:
            self.conv1 = conv(inputs, outputs, kernel_size, self.resample, self.padding, dilation=dilation, bias=False)
        else:
            self.conv1 = conv(inputs, outputs, kernel_size, 1, self.padding, dilation=dilation, bias=False)
            self.pool1 = maxpool(self.resample,self.resample) if self.pool=='Max' else avgpool(self.resample,self.resample)
        if self.inputs_resample or self.dim_missmatch:
            if pool is None:
                self.convskip = conv(inputs, outputs, kernel_size, self.resample, self.padding, dilation=dilation, bias=False)
            else:
                self.convskip = conv(inputs, outputs, kernel_size, 1, self.padding, dilation=dilation, bias=False)
                self.poolskip = maxpool(self.resample,self.resample) if self.pool=='Max' else avgpool(self.resample,self.resample)
                
        self.conv2 = conv(outputs, outputs, kernel_size, 1, self.padding, dilation=dilation, bias=False)
        self.norm2 = nn.GroupNorm(min(outputs,32),outputs)

    def forward(self,x):
        if self.residual:
            x = F.leaky_relu(self.norm1(x),0.2)
            if self.inputs_resample or self.dim_missmatch:
                # x_skip = F.avg_pool3d(x,self.resample,self.resample)
                x_skip = self.convskip(x)
                if self.pool is not None:
                    x_skip = self.poolskip(x_skip)
            else:
                x_skip = x
            x = F.leaky_relu(self.norm2(self.conv1(x)),0.2)
            if self.pool is not None:
                x = self.poolskip(x)
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
        self.conv1 = ECoGMappingBlock(16,32,[5,1,1],residual=True,resample = [2,1,1],pool='MAX')
        self.conv2 = ECoGMappingBlock(32,64,[3,1,1],residual=True,resample = [2,1,1],pool='MAX')
        self.norm_mask = nn.GroupNorm(32,64) 
        self.mask = ln.Conv3d(64,1,[3,1,1],1,[1,0,0])
        self.conv3 = ECoGMappingBlock(64,128,[3,3,3],residual=True,resample = [2,2,2],pool='MAX')
        self.conv4 = ECoGMappingBlock(128,256,[3,3,3],residual=True,resample = [2,2,2],pool='MAX')
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

    def forward(self,ecog,mask_prior,mni):
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


class BackBone(nn.Module):
    def __init__(self,attentional_mask=True):
        super(BackBone, self).__init__()
        self.attentional_mask = attentional_mask
        self.from_ecog = FromECoG(16,residual=True,shape='2D')
        self.conv1 = ECoGMappingBlock(16,32,[5,1],residual=True,resample = [1,1],shape='2D')
        self.conv2 = ECoGMappingBlock(32,64,[3,1],residual=True,resample = [1,1],shape='2D')
        self.norm_mask = nn.GroupNorm(32,64) 
        self.mask = ln.Conv2d(64,1,[3,1],1,[1,0])
    
    def forward(self,ecog):
        x_common_all = []
        mask_all=[]
        for d in range(len(ecog)):
            x = ecog[d]
            x = x.unsqueeze(1)
            x = self.from_ecog(x)
            x = self.conv1(x)
            x = self.conv2(x)
            if self.attentional_mask:
                mask = F.relu(self.mask(F.leaky_relu(self.norm_mask(x),0.2)))
                mask = mask[:,:,16:]
                x = x[:,:,16:]
                mask_all +=[mask]
            else:
                # mask = torch.sigmoid(self.mask(F.leaky_relu(self.norm_mask(x),0.2)))
                # mask = mask[:,:,16:]
                x = x[:,:,16:]
                # x = x*mask
                
            x_common_all +=[x]
            
        x_common = torch.cat(x_common_all,dim=0)
        if self.attentional_mask:
            mask = torch.cat(mask_all,dim=0)
        return x_common,mask.squeeze(1) if self.attentional_mask else None

class ECoGEncoderFormantHeads(nn.Module):
    def __init__(self,inputs,n_mels,n_formants):
        super(ECoGEncoderFormantHeads,self).__init__()
        self.n_mels = n_mels
        self.f0 = ln.Conv1d(inputs,1,1)
        self.loudness = ln.Conv1d(inputs,1,1)
        self.amplitudes = ln.Conv1d(inputs,2,1)
        self.freq_formants = ln.Conv1d(inputs,n_formants,1)
        self.bandwidth_formants = ln.Conv1d(inputs,n_formants,1)
        self.amplitude_formants = ln.Conv1d(inputs,n_formants,1)

    def forward(self,x):
        loudness = F.relu(self.loudness(x))
        f0 = F.sigmoid(self.f0(x)) * (15/64)*(self.n_mels/64) # 179hz < f0 < 420 hz
        # f0 = F.sigmoid(self.conv_f0(x_fundementals)) * (22/64)*(self.n_mels/64) - (16/64)*(self.n_mels/64)# 72hz < f0 < 253 hz, human voice
        # f0 = F.sigmoid(self.conv_f0(x_fundementals)) * (11/64)*(self.n_mels/64) - (-2/64)*(self.n_mels/64)# 160hz < f0 < 300 hz, female voice
        amplitudes = F.softmax(self.amplitudes(x),dim=1)
        freq_formants = F.sigmoid(self.freq_formants(x))
        freq_formants = torch.cumsum(freq_formants,dim=1)
        bandwidth_formants = F.sigmoid(self.bandwidth_formants(x))
        amplitude_formants = F.softmax(self.amplitude_formants(x),dim=1)
        return {'f0':f0,
                'loudness':loudness,
                'amplitudes':amplitudes,
                'freq_formants':freq_formants,
                'bandwidth_formants':bandwidth_formants,
                'amplitude_formants':amplitude_formants,}

@ECOG_ENCODER.register("ECoGMappingTransformer")
class ECoGMapping_Transformer(nn.Module):
    def __init__(self,n_mels,n_formants,SeqLen=128,hidden_dim=256,dim_feedforward=256,encoder_only=False,attentional_mask=False,n_heads=1,non_local=False):
        super(ECoGMapping_Transformer, self).__init__()
        self.n_mels = n_mels,
        self.n_formant = n_formants,
        self.encoder_only = encoder_only,
        self.attentional_mask = attentional_mask,
        self.backbone = BackBone(attentional_mask=attentional_mask)
        self.position_encoding = build_position_encoding(SeqLen,hidden_dim,'MNI')
        self.input_proj = ln.Conv2d(64, hidden_dim, kernel_size=1)
        if non_local:
            Transformer = TransformerNL
        else:
            Transformer = TransformerTS
        self.transformer = Transformer(d_model=hidden_dim, nhead=n_heads, num_encoder_layers=6,
                                    num_decoder_layers=6, dim_feedforward=dim_feedforward, dropout=0.1,
                                    activation="relu", normalize_before=False,
                                    return_intermediate_dec=False,encoder_only = encoder_only)
        self.output_proj = ECoGEncoderFormantHeads(hidden_dim,n_mels,n_formants)
        self.query_embed = nn.Embedding(SeqLen, hidden_dim)
    
    def forward(self,x,mask_prior,mni):
        features,mask = self.backbone(x)
        pos = self.position_encoding(mni)
        hs = self.transformer(self.input_proj(features), mask if self.attentional_mask else None, self.query_embed.weight, pos)
        if not self.encoder_only:
            hs,encoded = hs
            out = self.output_proj(hs)
        else:
            _,encoded = hs
            encoded = encoded.max(-1)[0]
            out = self.output_proj(encoded)
        return out
        


