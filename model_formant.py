import random
import losses
from net_formant import *
import numpy as np
class Model(nn.Module):
    def __init__(self, generator="", encoder="", ecog_encoder_name="",
                 spec_chans = 128, n_formants=2, with_ecog = False,
                 hidden_dim=256,dim_feedforward=256,encoder_only=True,attentional_mask=False,n_heads=1,non_local=False):
        super(Model, self).__init__()
        self.spec_chans = spec_chans
        self.with_ecog = with_ecog
        self.ecog_encoder_name = ecog_encoder_name
        self.decoder = GENERATORS[generator](
            n_mels = spec_chans,
            k = 30,
        )
        self.encoder = ENCODERS[encoder](
            n_mels = spec_chans,
            n_formants = n_formants,
        )
        if with_ecog:
            if 'Transformer' in ecog_encoder_name:
                self.ecog_encoder = ECOG_ENCODER[ecog_encoder_name](
                    n_mels = spec_chans,n_formants = n_formants,
                    hidden_dim=hidden_dim,dim_feedforward=dim_feedforward,n_heads=n_heads,
                    encoder_only=encoder_only,attentional_mask=attentional_mask,non_local=non_local,
                )
            else:
                self.ecog_encoder = ECOG_ENCODER[ecog_encoder_name](
                    n_mels = spec_chans,n_formants = n_formants,
                )

    def generate_fromecog(self, ecog = None, mask_prior = None, mni=None,return_components=False):
        components = self.ecog_encoder(ecog, mask_prior,mni)
        rec = self.decoder.forward(components)
        if return_components:
            return rec, components
        else:
            return rec

    def generate_fromspec(self, spec, return_components=False):
        components = self.encoder(spec)
        rec = self.decoder.forward(components)
        if return_components:
            return rec, components
        else:
            return rec

    def encode(self, spec):
        components = self.encoder(spec)
        return components
    
    def forward(self, spec, ecog, mask_prior, on_stage, ae, tracker, encoder_guide, mni=None):
        if ae:
            self.encoder.requires_grad_(True)
            rec = self.generate_fromspec(spec)
            Lae = torch.mean((rec - spec).abs())
            tracker.update(dict(Lae=Lae))
            return Lae
        else:
            self.encoder.requires_grad_(False)
            rec,components_ecog = self.generate_fromecog(ecog,mask_prior,mni=mni,return_components=True)
            Lrec = torch.mean((rec - spec).abs())
            tracker.update(dict(Lrec=Lrec))
            Lcomp = 0
            if encoder_guide:
                components_guide = self.encode(spec)   
                consonant_weight = 1#100*(torch.sign(components_guide['amplitudes'][:,1:]-0.5)*0.5+0.5)
                for key in components_guide.keys():
                    if key == 'loudness':
                        diff = torch.mean((components_guide[key] - components_ecog[key])**2) #+ torch.mean((components_guide[key] - components_ecog[key])**2 * on_stage * consonant_weight)
                    elif key in ['freq_formants', 'bandwidth_formants', 'amplitude_formants']:
                        diff = torch.mean((components_guide[key] - components_ecog[key])**2 * on_stage * consonant_weight)
                    else:
                        diff = torch.mean((components_guide[key] - components_ecog[key])**2 * on_stage)
                    tracker.update({key : diff})
                    Lcomp += diff
            
            Loss = Lrec+Lcomp
            return Loss

    def lerp(self, other, betta,w_classifier=False):
        if hasattr(other, 'module'):
            other = other.module
        with torch.no_grad():
            params = list(self.decoder.parameters()) + list(self.encoder.parameters()) + (list(self.ecog_encoder.parameters()) if self.with_ecog else [])
            other_param = list(other.decoder.parameters()) + list(other.encoder.parameters()) + (list(other.ecog_encoder.parameters()) if self.with_ecog else [])
            for p, p_other in zip(params, other_param):
                p.data.lerp_(p_other.data, 1.0 - betta)

