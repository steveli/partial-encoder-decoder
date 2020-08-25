import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import flow


class ConvEncoder(nn.Module):
    def __init__(self, latent_size, flow_depth=2, logprob=False):
        super().__init__()

        if logprob:
            self.encode_func = self.encode_logprob
        else:
            self.encode_func = self.encode

        DIM = 64
        self.main = nn.Sequential(
            nn.Conv2d(1, DIM, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(DIM, 2 * DIM, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(2 * DIM, 4 * DIM, 5, stride=2, padding=2),
            nn.ReLU(True),
        )

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
        conv_out_size = 4 * 4 * 4 * DIM
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, fc_out_size),
            nn.LayerNorm(fc_out_size),
            nn.LeakyReLU(0.2),
            nn.Linear(fc_out_size, fc_out_size),
        )

    def forward(self, input, k_samples=5):
        return self.encode_func(input, k_samples)

    def encode_logprob(self, input, k_samples=5):
        x = self.main(input.view(-1, 1, 28, 28))
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
        x = self.main(input.view(-1, 1, 28, 28))
        x = x.view(input.shape[0], -1)
        fc_out = self.fc(x).chunk(self.enc_chunk, dim=1)
        mu, logvar = fc_out[:2]
        std = F.softplus(logvar)
        qz_x = Normal(mu, std)
        z = qz_x.rsample()
        if self.q_z_flow:
            z, _ = self.q_z_flow(z, context=fc_out[2])
        return z
