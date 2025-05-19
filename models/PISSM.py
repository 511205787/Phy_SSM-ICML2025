import models.PISSM_pendulum_friction as PISSM_pendulum_friction
import models.PISSM_SIR as PISSM_SIR
import models.PISSM_drone as PISSM_drone
import torch.nn as nn
import torch
import torch.nn.functional as F
from s5 import S5
import torch.nn.functional as F
from typing import Tuple, Optional, Literal, List
Initialization = Literal['dense_columns', 'dense', 'factorized']

class GEGLU(torch.nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

class S5Extended(S5):
    def __init__(self, 
                 width: int, 
                 state_width: Optional[int] = None, 
                 factor_rank: Optional[int] = None, 
                 block_count: int = 1, 
                 dt_min: float = 0.001, 
                 dt_max: float = 0.1, 
                 liquid: bool = False, 
                 degree: int = 1, 
                 bidir: bool = False, 
                 bcInit: Optional[str] = None):
        super().__init__(width, state_width, factor_rank, block_count, dt_min, dt_max, liquid, degree, bidir, bcInit)

    def forward_rnn(self, signal, state, step_scale: float | torch.Tensor = 1.0):
        return self.seq.forward_rnn(signal, state, step_scale)

class S5Block(nn.Module):
    def __init__(self, dim: int, state_dim: int, bidir: bool, block_count: int = 1, liquid: bool = False, degree: int = 1, factor_rank: int | None = None, bcInit: Optional[str] = None, ff_mult: float = 1., glu: bool = True,
                 ff_dropout: float = 0.0, attn_dropout: float = 0.0):
        super().__init__()
        self.s5 = S5Extended(dim, state_width=state_dim, bidir=bidir, block_count=block_count, liquid=liquid, degree=degree, factor_rank=factor_rank, bcInit=bcInit)
        self.attn_norm = nn.LayerNorm(dim)
        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.geglu = GEGLU() if glu else None
        self.ff_enc = nn.Linear(dim, int(dim * ff_mult) * (1 + glu), bias=False)
        self.ff_dec = nn.Linear(int(dim * ff_mult), dim, bias=False)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff_dropout = nn.Dropout(p=ff_dropout)

    def initial_state(self, batch_size: Optional[int] = None):
        return self.s5.initial_state(batch_size)

    def forward(self, x, step_scale: float | torch.Tensor = 1.0, state=None, return_state=False):
        fx = self.attn_norm(x)
        res = fx.clone()
        x, next_state = self.s5(fx, step_scale=step_scale, state=state, return_state=return_state)
        x = F.gelu(x) + res
        x = self.attn_dropout(x)

        fx = self.ff_norm(x)
        res = fx.clone()
        x = self.ff_enc(fx)
        if self.geglu is not None:
            x = self.geglu(x)
        x = self.ff_dec(x) + res
        x = self.ff_dropout(x)

        if return_state:
            return x, next_state
        return x

    def forward_rnn(self, x, state, step_scale: float | torch.Tensor = 1.0):
        fx = self.attn_norm(x)
        res = fx.clone()
        x, state = self.s5.forward_rnn(fx, state, step_scale)
        x = F.gelu(x) + res
        x = self.attn_dropout(x)

        fx = self.ff_norm(x)
        res = fx.clone()
        x = self.ff_enc(fx)
        if self.geglu is not None:
            x = self.geglu(x)
        x = self.ff_dec(x) + res
        x = self.ff_dropout(x)

        return x, state
    
class MultilayerS5Block(nn.Module):
    def __init__(self, layers: List[S5Block]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def initial_state(self, batch_size: Optional[int] = None):
        return [layer.initial_state(batch_size) for layer in self.layers]

    def forward(self, x, step_scale: float | torch.Tensor = 1.0, state=None, return_state=False):
        if state is None:
            state = [None] * len(self.layers)
        next_states = []
        for i, layer in enumerate(self.layers):
            x, next_state = layer(x, step_scale=step_scale, state=state[i], return_state=True)
            next_states.append(next_state)
        if return_state:
            return x, next_states
        return x

    def forward_rnn(self, x, state, step_scale: float | torch.Tensor = 1.0):
        next_states = []
        for i, layer in enumerate(self.layers):
            x, next_state = layer.forward_rnn(x, state[i], step_scale)
            next_states.append(next_state)
        return x, next_states

class PISSM(nn.Module):
    def __init__(self, input_dim, ode_dim, ssm_input_dim, ssm_dropout, num_x2z_layers,
                 dim_pc_ssm, dim_control, n_ssmlayer_A, n_ssmlayer_B, extention, encoder, decoder):
        super(PISSM, self).__init__()
        self.dim_control = dim_control
        self.encoder = encoder(input_dim, ode_dim, ssm_input_dim, ssm_dropout)
        multi_x2z_layers = []
        for _ in range(num_x2z_layers):
            multi_x2z_layers.append(S5Block(ssm_input_dim, ssm_input_dim, bidir=False))
        self.x2z_layers = MultilayerS5Block(multi_x2z_layers)
        self.decoder = decoder(input_dim, dim_pc_ssm, ssm_input_dim, dim_control, ssm_dropout, n_ssmlayer_A, n_ssmlayer_B,
                               extention)
        self.prior_mean = nn.Linear(dim_pc_ssm, 16)
        self.prior_std = nn.Linear(dim_pc_ssm, 16)
        self.post_mean = nn.Linear(ssm_input_dim, 16)
        self.post_std = nn.Linear(ssm_input_dim, 16)
        # Latent vector to ODE input vector
        self.addf_layer = nn.Linear(ssm_input_dim, 3)

    def compute_time_intervals(self, t):
        # compute differences between successive time steps: shape [batch, seq-1]
        intervals = torch.diff(t, dim=1)
        
        # extract the very first interval for each batch
        # intervals[:, :1] has shape [batch, 1]
        first_interval = intervals[:, :1]
        
        # pad at the front with that same first interval
        intervals = torch.cat([first_interval, intervals], dim=1)
        return intervals
    
    def _reparameterized_sample(self, mu, std):
        eps = torch.empty(size=std.size(), device=mu.device, dtype=torch.float).normal_()
        return eps.mul(std).add(mu)

    def forward(self, mini_batch, control_data, t):
        intervals = self.compute_time_intervals(t)
        _, total_seq = intervals.size(0), intervals.size(1)
        _, seq, _ = mini_batch.size(0), mini_batch.size(1), mini_batch.size(2)
        observation = self.encoder(mini_batch, intervals[:, :seq])  # [batch, seq, dim]
        z_post = self.x2z_layers(observation, step_scale=intervals[:, :seq])
        latent_z_post_mean = self.post_mean(z_post)
        latent_z_post_std = self.post_std(z_post)
        latent_z = self._reparameterized_sample(latent_z_post_mean, latent_z_post_std)
        #
        batch_size, seq_len, dim_state = observation.shape
        # Decoder
        control_data = control_data.view(batch_size, total_seq, -1) # (batch, seq) -> (batch, seq ,1)
        predicted_batch, z_obs, z_generate = self.decoder(latent_z, control_data, intervals)
        latent_z_prior_mean = self.prior_mean(z_generate)
        latent_z_prior_std = self.prior_std(z_generate)
        return predicted_batch, z_obs, z_generate, (latent_z_post_mean, latent_z_post_std), (latent_z_prior_mean, latent_z_prior_std)


def create_pissm_pendulum_friction(input_dim=[28, 28], ode_dim=2, ssm_input_dim=32, ssm_dropout=0.1, num_x2z_layers=5, dim_pc_ssm=4, dim_control=1,
                          n_ssmlayer_A=3, n_ssmlayer_B=2, extention=True):
    return PISSM(input_dim, ode_dim, ssm_input_dim, ssm_dropout, num_x2z_layers, dim_pc_ssm, 
                 dim_control, n_ssmlayer_A, n_ssmlayer_B, extention, PISSM_pendulum_friction.Encoder, PISSM_pendulum_friction.Decoder)

def create_pissm_SIR(input_dim=3, ode_dim=3, ssm_input_dim=32, ssm_dropout=0.1, num_x2z_layers=4, dim_pc_ssm=3, dim_control=None,
                          n_ssmlayer_A=3, n_ssmlayer_B=2, extention=False):
    return PISSM(input_dim, ode_dim, ssm_input_dim, ssm_dropout, num_x2z_layers, dim_pc_ssm, 
                 dim_control, n_ssmlayer_A, n_ssmlayer_B, extention, PISSM_SIR.Encoder, PISSM_SIR.Decoder)

def create_pissm_drone(input_dim=9, ode_dim=10, ssm_input_dim=32, ssm_dropout=0.1, num_x2z_layers=5, dim_pc_ssm=11, dim_control=7,
                          n_ssmlayer_A=4, n_ssmlayer_B=None, extention=True):
    return PISSM(input_dim, ode_dim, ssm_input_dim, ssm_dropout, num_x2z_layers, dim_pc_ssm, 
                 dim_control, n_ssmlayer_A, n_ssmlayer_B, extention, PISSM_drone.Encoder, PISSM_drone.Decoder)