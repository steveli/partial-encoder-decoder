import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import flow


def conv_ln_lrelu(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, 5, 2, 2),
        nn.InstanceNorm2d(out_dim, affine=True),
        nn.LeakyReLU(0.2))


class ConvEncoder(nn.Module):
    def __init__(self, latent_size, flow_depth=2, logprob=False):
        super().__init__()

        if logprob:
            self.encode_func = self.encode_logprob
        else:
            self.encode_func = self.encode

        dim = 64
        self.ls = nn.Sequential(
            nn.Conv2d(3, dim, 5, 2, 2), nn.LeakyReLU(0.2),
            conv_ln_lrelu(dim, dim * 2),
            conv_ln_lrelu(dim * 2, dim * 4),
            conv_ln_lrelu(dim * 4, dim * 8),
            nn.Conv2d(dim * 8, latent_size, 4))

        if flow_depth > 0:
            # IAF
            hidden_size = latent_size * 2
            flow_layers = [flow.InverseAutoregressiveFlow(
                latent_size, hidden_size, latent_size)
                for _ in range(flow_depth)]

            flow_layers.append(flow.Reverse(latent_size))
            self.q_z_flow = flow.FlowSequential(*flow_layers)
            self.enc_chunk = 3
        else:
            self.q_z_flow = None
            self.enc_chunk = 2

        fc_out_size = latent_size * self.enc_chunk
        self.fc = nn.Sequential(
            nn.Linear(latent_size, fc_out_size),
            nn.LayerNorm(fc_out_size),
            nn.LeakyReLU(0.2),
            nn.Linear(fc_out_size, fc_out_size),
        )

    def forward(self, input, k_samples=5):
        return self.encode_func(input, k_samples)

    def encode_logprob(self, input, k_samples=5):
        x = self.ls(input)
        x = x.view(input.shape[0], -1)
        fc_out = self.fc(x).chunk(self.enc_chunk, dim=1)
        mu, logvar = fc_out[:2]
        std = F.softplus(logvar)
        qz_x = Normal(mu, std)
        z = qz_x.rsample([k_samples])
        log_q_z = qz_x.log_prob(z)
        if self.q_z_flow:
            z, log_q_z_flow = self.q_z_flow(z, context=fc_out[2])
            log_q_z = (log_q_z + log_q_z_flow).sum(-1)
        else:
            log_q_z = log_q_z.sum(-1)
        return z, log_q_z

    def encode(self, input, _):
        x = self.ls(input)
        x = x.view(input.shape[0], -1)
        fc_out = self.fc(x).chunk(self.enc_chunk, dim=1)
        mu, logvar = fc_out[:2]
        std = F.softplus(logvar)
        qz_x = Normal(mu, std)
        z = qz_x.rsample()
        if self.q_z_flow:
            z, _ = self.q_z_flow(z, context=fc_out[2])
        return z
