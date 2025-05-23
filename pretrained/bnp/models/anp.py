import torch
import torch.nn as nn
from torch.distributions import kl_divergence
import attridict

from utils.misc import stack, logmeanexp
from utils.sampling import sample_subset

from models.modules import CrossAttnEncoder, PoolingEncoder, Decoder, PositionEmbedder
import pdb 

class ANP(nn.Module):
    def __init__(self,
            emb, 
            dim_x=1,
            dim_y=1,
            dim_hid=128,
            dim_lat=128,
            enc_v_depth=4,
            enc_qk_depth=2,
            enc_pre_depth=4,
            enc_post_depth=2,
            dec_depth=3):

        super().__init__()

        if emb is None:
            dim_x_aug = dim_x 
            self.emb = nn.Identity()
        else:
            dim_x_aug = emb.n_latents
            self.emb = emb 

        self.denc = CrossAttnEncoder(
                dim_x=dim_x_aug,
                dim_y=dim_y,
                dim_hid=dim_hid,
                v_depth=enc_v_depth,
                qk_depth=enc_qk_depth)

        self.lenc = PoolingEncoder(
                dim_x=dim_x_aug,
                dim_y=dim_y,
                dim_hid=dim_hid,
                dim_lat=dim_lat,
                self_attn=True,
                pre_depth=enc_pre_depth,
                post_depth=enc_post_depth)

        self.dec = Decoder(
                dim_x=dim_x_aug,
                dim_y=dim_y,
                dim_enc=dim_hid+dim_lat,
                dim_hid=dim_hid,
                depth=dec_depth)

    def predict(self, xc, yc, xt, z=None, num_samples=None):
        xc = self.emb(xc)
        xt = self.emb(xt)
        theta = stack(self.denc(xc, yc, xt), num_samples)
        if z is None:
            pz = self.lenc(xc, yc)
            z = pz.rsample() if num_samples is None \
                    else pz.rsample([num_samples])
        z = stack(z, xt.shape[-2], -2)
        encoded = torch.cat([theta, z], -1)
        return self.dec(encoded, stack(xt, num_samples))

    def forward(self, batch, num_samples=None, reduce_ll=True):
        outs = attridict()
        if self.training:
            pz = self.lenc(self.emb(batch.xc), batch.yc)
            qz = self.lenc(self.emb(batch.x), batch.y)
            z = qz.rsample() if num_samples is None else \
                    qz.rsample([num_samples])
            py = self.predict(batch.xc, batch.yc, batch.x,
                    z=z, num_samples=num_samples)

            if num_samples > 1:
                # K * B * N
                recon = py.log_prob(stack(batch.y, num_samples)).sum(-1)

                # Compute FFT error
                dx = (batch.x[:,1,:]-batch.x[:,0,:]).mean()
                freqs = torch.fft.rfftfreq(batch.y.shape[-2], d=dx)
                true_fft = torch.fft.rfft(stack(batch.y, num_samples), dim=2)
                pred_fft = torch.fft.rfft(py.mean, dim=2)
                fft_mse = ((pred_fft - true_fft)**2).sum(-1).abs()
                # K * B
                log_qz = qz.log_prob(z).sum(-1)
                log_pz = pz.log_prob(z).sum(-1)

                # K * B
                log_w = recon.sum(-1) + log_pz - log_qz
                t1 = -logmeanexp(log_w).mean()/ batch.x.shape[-2]
                t2 = fft_mse.mean()
                alpha = 1.0
                outs.loss = alpha*t1 + (1-alpha)*t2
            else:
                outs.recon = py.log_prob(batch.y).sum(-1).mean()
                outs.kld = kl_divergence(qz, pz).sum(-1).mean()
                outs.loss = -outs.recon + outs.kld / batch.x.shape[-2]

        else:
            py = self.predict(batch.xc, batch.yc, batch.x, num_samples=num_samples)
            if num_samples is None:
                ll = py.log_prob(batch.y).sum(-1)
                mse = ((py.mean-batch.y)**2).sum(-1)
            else:
                y = torch.stack([batch.y]*num_samples)
                if reduce_ll:
                    ll = logmeanexp(py.log_prob(y).sum(-1))
                    mse = ((py.mean-y)**2).sum(-1).mean()
                else:
                    ll = py.log_prob(y).sum(-1)
                    mse = ((py.mean-y)**2).sum(-1)
            num_ctx = batch.xc.shape[-2]

            if reduce_ll:
                outs.mse = mse.mean()
                outs.ctx_ll = ll[...,:num_ctx].mean()
                outs.tar_ll = ll[...,num_ctx:].mean()
            else:
                out.mse = mse
                outs.ctx_ll = ll[...,:num_ctx]
                outs.tar_ll = ll[...,num_ctx:]

        return outs
