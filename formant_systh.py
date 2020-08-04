import torch
from torch import nn
# from torch.nn import functional as F
# from registry import *
import lreq as ln
import json
from tqdm import tqdm
import os
import numpy as np
from torch.nn import functional as F
from torchvision.utils import save_image
from torch.nn.parameter import Parameter
from custom_adam import LREQAdam
from ECoGDataSet import ECoGDataset


def freq_coloring(sample,ind,color='r'):
   for b in range(sample.shape[0]):
      for t in range(sample.shape[2]):
         if color == 'r':
            sample[b,0,t,ind[b,0,t]]=1
         if color == 'g':
            sample[b,1,t,ind[b,0,t]]=1
         if color == 'b':
            sample[b,2,t,ind[b,0,t]]=1
         if color == 'y':
            sample[b,0,t,ind[b,0,t]]=1;sample[b,1,t,ind[b,0,t]]=1
         if color == 'c':
            sample[b,1,t,ind[b,0,t]]=1;sample[b,2,t,ind[b,0,t]]=1
         if color == 'm':
            sample[b,0,t,ind[b,0,t]]=1;sample[b,2,t,ind[b,0,t]]=1
   return sample

def voicing_coloring(sample,amplitude):
   for b in range(sample.shape[0]):
      for t in range(sample.shape[2]):
         sample[b,:,t]=sample[b,0:1,t]*(amplitude[b,0,t]*torch.tensor([0.,0.35,0.67]) + amplitude[b,1,t]*torch.tensor([1.0,0.87,0.])).unsqueeze(dim=-1) # voicing for blue, unvoicing for yellow
   return sample
         
def color_spec(spec,components,n_mels):
   clrs = ['g','y','b','m','c']
   sample_in = spec.repeat(1,3,1,1)
   sample_in = sample_in * 0.5 + 0.5
   f0=(components['f0']*n_mels).int().clamp(min=0,max=n_mels-1)
   formants_freqs=(components['freq_formants']*n_mels).int().clamp(min=0,max=n_mels-1)
   sample_in_color_voicing = sample_in.clone()
   sample_in_color_voicing = voicing_coloring(sample_in_color_voicing,components['amplitudes'])
   sample_in_color_freq = sample_in.clone()
   sample_in_color_freq = sample_in_color_freq/2
   sample_in_color_freq = freq_coloring(sample_in_color_freq,f0,'r')
   for j in range(formants_freqs.shape[1]):
      sample_in_color_freq = freq_coloring(sample_in_color_freq,formants_freqs[:,j].unsqueeze(1),clrs[j])
   return sample_in_color_voicing,sample_in_color_freq

def save_sample(sample,ecog,mask_prior,encoder,decoder,ecog_encoder,epoch,mode='test',path='training_artifacts/formantsysth_voicingandunvoicing_loudness',tracker=None):
   os.makedirs(path, exist_ok=True)
   with torch.no_grad():
      encoder.eval()
      decoder.eval()
      if ecog_encoder is not None:
         ecog_encoder.eval()
      sample_in_all = torch.tensor([])
      sample_in_color_freq_all = torch.tensor([])
      sample_in_color_voicing_all = torch.tensor([])
      rec_all = torch.tensor([])
      if ecog_encoder is not None:
         sample_in_color_freq_ecog_all = torch.tensor([])
         sample_in_color_voicing_ecog_all = torch.tensor([])
         rec_ecog_all = torch.tensor([])
      n_mels = sample.shape[-1]
      for i in range(0,sample.shape[0],9):
         sample_in = sample[i:np.minimum(i+9,sample.shape[0])]
         if ecog_encoder is not None:
            ecog_in = [ecog[j][i:np.minimum(i+9,sample.shape[0])] for j in range(len(ecog))]
            mask_prior_in = [mask_prior[j][i:np.minimum(i+9,sample.shape[0])] for j in range(len(ecog))]
         components = encoder(sample_in)
         rec = decoder(components)
         sample_in_color_voicing,sample_in_color_freq = color_spec(sample_in,components,n_mels)
         rec = rec.repeat(1,3,1,1)
         rec = rec * 0.5 + 0.5
         sample_in_all = torch.cat([sample_in_all,sample_in],dim=0)
         sample_in_color_freq_all = torch.cat([sample_in_color_freq_all,sample_in_color_freq],dim=0)
         sample_in_color_voicing_all = torch.cat([sample_in_color_voicing_all,sample_in_color_voicing],dim=0)
         rec_all = torch.cat([rec_all,rec],dim=0)
         if ecog_encoder is not None:
            components_ecog = ecog_encoder(ecog_in,mask_prior_in)
            rec_ecog = decoder(components_ecog)
            rec_ecog = rec_ecog.repeat(1,3,1,1)
            rec_ecog = rec_ecog * 0.5 + 0.5
            sample_in_color_voicing_ecog,sample_in_color_freq_ecog = color_spec(sample_in,components_ecog,n_mels)
            sample_in_color_freq_ecog_all = torch.cat([sample_in_color_freq_ecog_all,sample_in_color_freq_ecog],dim=0)
            sample_in_color_voicing_ecog_all = torch.cat([sample_in_color_voicing_ecog_all,sample_in_color_voicing_ecog],dim=0)
            rec_ecog_all = torch.cat([rec_ecog_all,rec_ecog],dim=0)
      sample_in_all = sample_in_all.repeat(1,3,1,1)*0.5+0.5
      if ecog_encoder is None:
         resultsample = torch.cat([sample_in_all, sample_in_color_freq_all,sample_in_color_voicing_all, rec_all], dim=0)
      else:
         resultsample = torch.cat([sample_in_all, sample_in_color_freq_all,sample_in_color_voicing_all, rec_all,sample_in_color_freq_ecog_all,sample_in_color_voicing_ecog_all, rec_ecog_all], dim=0)
      resultsample = resultsample.transpose(-2,-1)
      resultsample = resultsample.cpu()
      # import pdb;pdb.set_trace()
      if mode == 'train':
         f = os.path.join(path,'sample_train_%d.png' % (epoch + 1))
      if mode == 'test':
         f = os.path.join(path,'sample_%d.png' % (epoch + 1))
      save_image(resultsample, f, nrow=resultsample.shape[0]//(4 if ecog_encoder is None else 7))
      if mode == 'test':
         tracker.register_means(epoch)
      return

def main():
   OUTPUT_DIR = 'training_artifacts/formantsysth_voicingandunvoicing_loudness_NY742'
   LOAD_DIR = ''
   # LOAD_DIR = 'training_artifacts/formantsysth_voicingandunvoicing_loudness_'
   torch.set_default_tensor_type('torch.cuda.FloatTensor')
   device = torch.device("cuda:0")
   encoder = FormantEncoder(n_mels=64,n_formants=2,k=30)
   decoder = FormantSysth(n_mels=64,k=30)
   encoder.cuda()
   decoder.cuda()
   if LOAD_DIR is not '':
      encoder.load_state_dict(torch.load(os.path.join(LOAD_DIR,'encoder_60.pth')))
      decoder.load_state_dict(torch.load(os.path.join(LOAD_DIR,'decoder_60.pth')))
   optimizer = LREQAdam([
            {'params': encoder.parameters()},
            {'params': decoder.parameters()}
        ], lr=0.01, weight_decay=0)
   with open('train_param.json','r') as rfile:
        param = json.load(rfile)
   dataset = torch.utils.data.DataLoader(ECoGDataset(['NY742'],mode='train'),batch_size=32,shuffle=True, drop_last=True)
   dataset_test = torch.utils.data.DataLoader(ECoGDataset(['NY742'],mode='test'),batch_size=50,shuffle=False, drop_last=False)
   sample_dict_test = next(iter(dataset_test))
   sample_spec_test = sample_dict_test['spkr_re_batch_all'].to('cuda').float()
   for epoch in range(60):
      encoder.train()
      decoder.train()
      for sample_dict_train in tqdm(iter(dataset)):
         x = sample_dict_train['spkr_re_batch_all'].to('cuda').float()
         optimizer.zero_grad()
         f0,loudness,amplitudes,formants_freqs,formants_bandwidth,formants_amplitude = encoder(x)
         x_rec = decoder(f0,loudness,amplitudes,formants_freqs,formants_bandwidth,formants_amplitude)
         loss = torch.mean((x-x_rec).abs())
         loss.backward()
         optimizer.step()
      save_sample(sample_spec_test,encoder,decoder,epoch,mode='test',path=OUTPUT_DIR)
      save_sample(x,encoder,decoder,epoch,mode='train',path=OUTPUT_DIR)
      torch.save(encoder.state_dict(),os.path.join(OUTPUT_DIR,'encoder_%d.pth' % (epoch+1)))
      torch.save(decoder.state_dict(),os.path.join(OUTPUT_DIR,'decoder_%d.pth' % (epoch+1)))

      
if __name__ == "__main__":
   main()
