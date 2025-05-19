import torch.nn as nn
import torch
import torch.nn.functional as F
from s5 import S5
import torch.nn.functional as F
from typing import Tuple, Optional, Literal, List
Initialization = Literal['dense_columns', 'dense', 'factorized']

def kld_gauss(mean_1, std_1, mean_2, std_2, masks=None, sum=False):
    EPS = torch.finfo(torch.float).eps
    """Using std to compute KLD"""
    kld_element =  (2 * torch.log(std_2 + EPS) - 2 * torch.log(std_1 + EPS) +
        (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
        std_2.pow(2) - 1)
    if masks is None:
        if sum:
            loss = 0.5 * torch.sum(kld_element.flatten(1), dim=1)
        else:
            loss = 0.5 * torch.mean(kld_element.flatten(1), dim=1)
    else:
        masks =(masks.sum(-1)>0)
        masks_sum = masks.sum(-1)
        masks_sum[masks_sum==0] = 1
        if sum:
            loss = 0.5 * torch.sum((kld_element * masks[..., None]).sum(-1), dim=-1)
        else:
            loss =	0.5 * torch.sum((kld_element * masks[...,None]).mean(-1), dim=-1) / masks_sum
    return loss

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

class Encoder(nn.Module):
    def __init__(self, input_dim, ssm_input_dim):
        '''
        Encoder here is used to lift the input data to the SSM input space. 
        You can use a simple MLP or a more complex architecture, e.g CNN for image, or GNN for graph.
        '''
        super(Encoder, self).__init__()
        self.lift_layer = nn.Linear(input_dim, ssm_input_dim)
        self.relu = nn.ReLU()

    def forward(self, input_batch):
        out = self.relu(self.lift_layer(input_batch))
        return out   # observation

class Decoder(nn.Module):
    def __init__(self, input_dim, dim_pc_ssm):
        '''
        Decoder here is used to lift the SSM output data to the original input space. 
        You can use a simple MLP or a more complex architecture, e.g CNN for image, or GNN for graph.
        '''
        super(Decoder, self).__init__()
        self.project_layer = nn.Linear(dim_pc_ssm, input_dim)

    def forward(self, input_batch):
        out = self.project_layer(input_batch)
        return out
###############
class SSM_step_ODE(nn.Module):
    def __init__(
        self,
        dim_control: Optional[int],
        dim_pc_ssm: int,
        A_known_const: Optional[torch.Tensor] = None,
        B_known_const: Optional[torch.Tensor] = None,
        mask_A: Optional[torch.Tensor]       = None,
        mask_B: Optional[torch.Tensor]       = None,
    ):
        """
        dim_pc_ssm:      the state dimension N
        dim_control:     the control dimension M (or None)
        A_known_const:   Tensor[N, N] or None -> default zeros
        B_known_const:   Tensor[N, M] or None -> default zeros (if dim_control)
        mask_A:          Tensor[N, N] or None -> default ones
        mask_B:          Tensor[N, M] or None -> default ones (if dim_control)
        """
        super().__init__()
        self.dim_control = dim_control
        N = dim_pc_ssm
        M = dim_control

        # ─── shape assertions ─────────────────────────────────────────
        if A_known_const is not None:
            assert A_known_const.shape == (N, N), \
                f"A_known_const must be ({N},{N}), got {tuple(A_known_const.shape)}"
        if mask_A is not None:
            assert mask_A.shape == (N, N), \
                f"mask_A must be ({N},{N}), got {tuple(mask_A.shape)}"
        if dim_control is not None:
            if B_known_const is not None:
                assert B_known_const.shape == (N, M), \
                    f"B_known_const must be ({N},{M}), got {tuple(B_known_const.shape)}"
            if mask_B is not None:
                assert mask_B.shape == (N, M), \
                    f"mask_B must be ({N},{M}), got {tuple(mask_B.shape)}"
        # ───────────────────────────────────────────────────────────────
                
        # defaults:
        if A_known_const is None:
            A_known_const = torch.zeros(N, N)
        if mask_A is None:
            mask_A = torch.ones(N, N)
        if dim_control is not None:
            if B_known_const is None:
                B_known_const = torch.zeros(N, M)
            if mask_B is None:
                mask_B = torch.ones(N, dim_control)
        else:
            mask_B = None
            B_known_const = None

        # register as buffers so they live on the right device & dtype
        self.register_buffer('A_known_const', A_known_const.float())
        self.register_buffer('mask_A',       mask_A.float())
        if mask_B is not None:
            self.register_buffer('mask_B',   mask_B.float())
        if B_known_const is not None:
            self.register_buffer('B_known_const', B_known_const.float())

    def forward(self, Unk_A, Unk_B, z_input, u_input):
        if Unk_A.dim() == 3:
            return self.get_seq_dynamics(Unk_A, Unk_B, z_input, u_input)
        else:
            return self.get_dynamics(Unk_A, Unk_B, z_input, u_input)

    def get_dynamics(self, Unk_A, Unk_B, z_input, u_input):
        batch, N = z_input.size(0), z_input.size(1)
        device = z_input.device

        # 1) known constant part
        #    expand to [batch, N, N]
        A_const = self.A_known_const.to(device).unsqueeze(0).expand(batch, -1, -1)

        # 2) known dynamic part(note that this part can be related to the time-variant input)
        A_dyn = torch.zeros(batch, N, N, device=device)
        # user defined part, see the example below
        # A_dyn[:, 2, 3] = z_input[:, 1]  # For pendulum example: set omega(t) at the right position
        # A_dyn[:, 3, 2] = -z_input[:, 1] # For pendulum example: set -omega(t) at the right position

        # combine
        A_known = A_const + A_dyn

        # 3) unknown part masked by mask_A
        maskA = self.mask_A.to(device).unsqueeze(0).expand(batch, -1, -1)
        A_unknown = Unk_A.view(batch, N, N) * maskA

        # 4) B part
        if self.dim_control is not None:
            maskB = self.mask_B.to(device).unsqueeze(0).expand(batch, -1, -1)
            B_unknown = Unk_B.view(batch, N, self.dim_control) * maskB
            B_const = self.B_known_const.to(device).unsqueeze(0).expand(batch, -1, -1)
            B = B_const + B_unknown
        else:
            B = None

        A = A_known + A_unknown
        return A, B

    def get_seq_dynamics(self, Unk_A, Unk_B, z_input, u_input):
        batch, seq, N = z_input.shape
        device = z_input.device

        # known constant part
        A_const = self.A_known_const.to(device)[None, None, :, :].expand(batch, seq, N, N)

        # known dynamic part(note that this part can be related to the time-variant input)
        A_dyn = torch.zeros(batch, seq, N, N, device=device)
        # user defined part, see the example below
        # A_dyn[:, :, 2, 3] = z_input[:, :, 1]    # For pendulum example: set omega(t) at the right position
        # A_dyn[:, :, 3, 2] = -z_input[:, :, 1]   # For pendulum example: set -omega(t) at the right position

        A_known = A_const + A_dyn

        # unknown A
        maskA = self.mask_A.to(device)[None, None, :, :].expand(batch, seq, N, N)
        A_unknown = Unk_A.view(batch, seq, N, N) * maskA

        # B part
        if self.dim_control is not None:
            maskB = self.mask_B.to(device)[None, None, :, :].expand(batch, seq, N, self.dim_control)
            B_unknown = Unk_B.view(batch, seq, N, self.dim_control) * maskB
            B_const = self.B_known_const.to(device)[None, None, :, :].expand(batch, seq, N, self.dim_control)
            B = B_const + B_unknown
        else:
            B = None

        A = A_known + A_unknown
        return A, B

class Physics_constraint_SSM(nn.Module):
    def __init__(
        self,
        dim_state=64,
        dim_control=None,
        n_ssmlayer_A=2,
        n_ssmlayer_B=2,
        A_known_const: Optional[torch.Tensor] = None,
        B_known_const: Optional[torch.Tensor] = None,
        mask_A: Optional[torch.Tensor] = None,
        mask_B: Optional[torch.Tensor] = None,
    ):
        super(Physics_constraint_SSM, self).__init__()
        self.dim_state = dim_state  # Number of original state dimensions of (u,x)
        self.dim_control = dim_control  # Number of control signal dimensions
        state_hidden = 128  # s5 hidden state dimension
        # A_unknown
        self.n_ssmlayer_A = n_ssmlayer_A
        multi_s5_A_layers = []

        for _ in range(n_ssmlayer_A):
            multi_s5_A_layers.append(S5Block(dim_state, state_hidden, bidir=False))

        self.ssm_A_layers = MultilayerS5Block(multi_s5_A_layers)
        
        if self.dim_control is not None:
            # B_unknown
            self.n_ssmlayer_B = n_ssmlayer_B
            multi_s5_B_layers = []

            for _ in range(n_ssmlayer_B):
                multi_s5_B_layers.append(S5Block(dim_state, state_hidden, bidir=False))

            self.ssm_B_layers = MultilayerS5Block(multi_s5_B_layers)
        else:
            self.ssm_B_layers = nn.Identity()

        self.ssm_to_A = nn.Sequential(nn.Linear(dim_state, state_hidden), nn.Softplus(),
                                      nn.Linear(state_hidden, dim_state * dim_state))
        if self.dim_control is not None:
            self.ssm_to_B = nn.Sequential(nn.Linear(dim_state, state_hidden), nn.Softplus(),
                                          nn.Linear(state_hidden, dim_state * self.dim_control))
        else:
            self.ssm_to_B = nn.Identity()

        self.SSM_Matrix = SSM_step_ODE(dim_control, dim_pc_ssm=dim_state,
                                        A_known_const=A_known_const,
                                        B_known_const=B_known_const,
                                        mask_A=mask_A,
                                        mask_B=mask_B)

    def bilinear(self, dt, A, B=None):
        if dt.dim() == 1:
            dt_expanded = dt.unsqueeze(-1).unsqueeze(-1)  # [seq_len, 1, 1] appropriate for broadcasting
        else:
            dt_expanded = dt.unsqueeze(-1).unsqueeze(-1)  # [..., 1, 1] appropriate for broadcasting

        N = A.shape[-1]  # Number of states
        M = B.shape[-1] if B is not None else None  # Number of control signal
        I = torch.eye(N).to(A)
        A_backwards = I - dt_expanded / 2 * A
        A_forwards = I + dt_expanded / 2 * A

        dA = torch.linalg.solve(A_backwards, A_forwards)  # (... N N)

        if B is None:
            dB = None
        else:
            dB_results = []
            for i in range(B.shape[-1]):
                Bi = B[..., i].unsqueeze(-1)
                dB_i = torch.linalg.solve(A_backwards, dt_expanded * Bi).squeeze(-1)
                dB_results.append(dB_i)
            dB = torch.stack(dB_results, dim=-1)

        return dA, dB

    def update_state(self, dA, dB, X, U):
        if dA.dim() == 4:  # Shape: [batch, seq, N, N]
            if dB is None:
                X_next = torch.einsum('bsij,bsj->bsi', dA, X)
            else:
                X_next = torch.einsum('bsij,bsj->bsi', dA, X) + torch.einsum('bsij,bsj->bsi', dB, U)
        elif dA.dim() == 3:  # Shape: [batch, N, N]
            if dB is None:
                X_next = torch.einsum('bij,bj->bi', dA, X)
            else:
                X_next = torch.einsum('bij,bj->bi', dA, X) + torch.einsum('bij,bj->bi', dB, U)
        return X_next
    
    def forward(self, interval, input_x, input_ux, control_data, **kwargs):
        """
        t: time; shape = torch.rand((seq_len))
        input_x: the system state; shape = torch.rand((batch_size, seq_len, N)) 
        input_u (optional): the control signal; shape = torch.rand((batch_size, seq_len, M)) 
        return the system's state; torch.rand((batch_size, seq_len, N)) 
        """

        batch_size, seq_len, dim_state = input_x.shape
        interval = interval[:, :seq_len]
        # Approximate the unknown system dynamics function by SSM
        Unk_A, next_state_z = self.ssm_A_layers(input_ux, step_scale=interval, return_state=True)
        Unk_A = self.ssm_to_A(Unk_A)
        if self.dim_control is not None:
            Unk_B, next_state_u = self.ssm_B_layers(input_ux, step_scale=interval, return_state=True)
            Unk_B = self.ssm_to_B(Unk_B)
        else:
            # dummy zero for shape inference or skip entirely
            Unk_B, next_state_u = None, None

        A, B = self.SSM_Matrix(Unk_A, Unk_B, input_x, control_data)
        # Discretize A and B matrices using bilinear transformation
        dA, dB = self.bilinear(interval, A, B)
        # Update state
        next_pred = self.update_state(dA, dB, input_x, control_data)
        return next_pred, next_state_z, next_state_u
    
    def step(self, interval, input_x, control_data, input_ux, state_z, state_u, **kwargs):
        """
        state_z: set up by default_state function
        state_u: if dim_control is not None, set up by default_state function, else 0
        input_x: (batch, d_input)
        input_ux: (batch, d_input), (u,x) latent
        control_data: (batch, d_input)
        Return: 
        next_pred: the next time step system state (batch, d_input)
        next_state_z, next_state_u: The SSM state.
        """

        batch_size, dim_state = input_x.shape
        # Approximate the unknown system dynamics function A_{unknown} by SSM
        Unk_A, next_state_z = self.ssm_A_layers.forward_rnn(input_ux, state_z, step_scale=interval)
        Unk_A = self.ssm_to_A(Unk_A)
        if self.dim_control is not None:
            Unk_B, next_state_u = self.ssm_B_layers.forward_rnn(input_ux, state_u, step_scale=interval)
            Unk_B = self.ssm_to_B(Unk_B)
        else:
            # dummy zero for shape inference or skip entirely
            Unk_B, next_state_u = None, None

        A, B = self.SSM_Matrix(Unk_A, Unk_B, input_x, control_data)
        # Discretize A and B matrices using bilinear transformation
        dA, dB = self.bilinear(interval, A, B)
        # Update state
        next_pred = self.update_state(dA, dB, input_x, control_data)
        return next_pred, next_state_z, next_state_u
    
class Phy_SSM_Unit(nn.Module):
    def __init__(self, input_dim, ode_dim, dim_pc_ssm, sample_hidden, ssm_input_dim, dim_control, n_ssmlayer_A, n_ssmlayer_B, extention,
                 A_known_const: Optional[torch.Tensor] = None, 
                 B_known_const: Optional[torch.Tensor] = None,
                 mask_A: Optional[torch.Tensor] = None,
                 mask_B: Optional[torch.Tensor] = None):
        '''
        input_dim: original input dimension
        dim_pc_ssm: dim state of Physics-constraint-SSM module, include the extention state
        '''
        super(Phy_SSM_Unit, self).__init__()
        self.input_dim = input_dim
        self.extention = extention  # Whether to extend the system state
        self.dim_control = dim_control
        self.relu = nn.ReLU()
        # Control Signal lift & merge
        if self.dim_control is not None:
            self.control_merge = nn.Sequential(nn.Linear(dim_control, ssm_input_dim), nn.Softplus(), nn.Linear(ssm_input_dim, dim_pc_ssm))
        else:
            self.control_merge = nn.Identity()
        # Physics_constraint_SSM
        self.Physics_constraint_SSM = Physics_constraint_SSM(dim_state=dim_pc_ssm, dim_control=dim_control, n_ssmlayer_A = n_ssmlayer_A, n_ssmlayer_B=n_ssmlayer_B, 
                                                             A_known_const=A_known_const, B_known_const=B_known_const, mask_A=mask_A, mask_B=mask_B)
        self.dim_pc_ssm = dim_pc_ssm 

        # Activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.latent2z = nn.Linear(sample_hidden, ode_dim)

    def forward(self, latent_z_batch, control_data, interval):
        batch_size, seq_len, dim_state = latent_z_batch.shape[0], latent_z_batch.shape[1], latent_z_batch.shape[2]
        _, total_len = interval.shape[0], interval.shape[1]
        device = latent_z_batch.device
        # # Lift the control data & z0
        # latent_control = self.control_merge(control_data)
        z = self.latent2z(latent_z_batch)
        # Expand the state space if necessary
        if self.extention:
            # Extend input_x with additional dimensions as needed
            # For example, adding sin and cos of the angle as new state variables
            # latent_batch[:, :, 0] contains theta (angle) and latent_batch[:, :, 1] contains omega (angular velocity)
            # We first compute the sin and cos of the angle
            sin_theta = torch.sin(z[:, :, 0])  # sin(theta(t))
            cos_theta = torch.cos(z[:, :, 0])  # cos(theta(t))
            # Now we concatenate these new variables with the original state variables to extend the state space
            z = torch.cat([z, sin_theta.unsqueeze(-1), cos_theta.unsqueeze(-1)], dim=-1)
        # 

        # get the (u,z) latent as SSM input
        if self.dim_control is not None:
            latent_control = self.control_merge(control_data)           # [B,seq,hidden]
            latent_uz = z + latent_control[:, :seq_len, :]
            cd_chunk = control_data[:, :seq_len]
        else:
            latent_control = None
            latent_uz = z
            cd_chunk = None
        z_state, state_z_ssm, state_u_ssm = self.Physics_constraint_SSM(interval[:, :seq_len], z, latent_uz, cd_chunk)
        z_obs = z
        z_generate = torch.cat([z[:, 0, :].unsqueeze(1), z_state[:, :-1, :]], dim=1)
        # 
        n_steps = total_len - seq_len
        z_extra = z_state[:, -1, :]
        z_extra_seq = torch.empty(batch_size, n_steps, z_extra.shape[1], device=device)
        for i in range(total_len - seq_len): 
            if self.dim_control is not None:
                u_i = control_data[:, seq_len+i]
                uz = z_extra + latent_control[:, seq_len+i] # merge (u,z)
            else:
                u_i = None
                uz = z_extra
            z_extra, state_z_ssm, state_u_ssm = self.Physics_constraint_SSM.step(interval[:, i + seq_len], z_extra, u_i, uz, state_z_ssm, state_u_ssm)
            z_extra_seq[:, i, :] = z_extra
        z_final = torch.cat([z_state, z_extra_seq], dim=1)

        return z_final, z_obs, z_generate
###############

class Phy_SSM(nn.Module):
    """
    Physics-Enhanced State Space Model (Phy_SSM).
    Args:
        input_dim (int): Dimension of the original input state (e.g., 2 for pendulum [theta, omega], or any observation dim).
        ode_dim (int): Dimension of the underlying ODE latent dim.
        ssm_input_dim (int): Dimension of inputs to the S5 x2z layers.
        num_x2z_layers (int): Number of S5Block layers mapping observations to latent z.
        dim_pc_ssm (int): Dimension of the Phy-SSM unit (include extended state, if no extension, should be same as ode_dim).
        dim_control (Optional[int]): Dimension of control input; if None, no control is used.
        n_ssmlayer_A (int): Number of S5Block layers modeling unknown A dynamics.
        n_ssmlayer_B (int): Number of S5Block layers modeling unknown B dynamics.
        extention (bool): Whether to extend the latent state with non-linear func. (e.g., sin/cos) (other complex non-linear func can also be used).
        A_known_const (Optional[Tensor]): Predefined constant part of A known matrix [dim_pc_ssm x dim_pc_ssm].
        B_known_const (Optional[Tensor]): Predefined constant part of B known matrix [dim_pc_ssm x dim_control].
        mask_A (Optional[Tensor]): Mask for unknown A entries [dim_pc_ssm x dim_pc_ssm].
        mask_B (Optional[Tensor]): Mask for unknown B entries [dim_pc_ssm x dim_control].
    """
    def __init__(self, input_dim, ode_dim, ssm_input_dim, num_x2z_layers,
                 dim_pc_ssm, dim_control, n_ssmlayer_A, n_ssmlayer_B, extention, 
                 A_known_const: Optional[torch.Tensor] = None, 
                 B_known_const: Optional[torch.Tensor] = None, 
                 mask_A: Optional[torch.Tensor] = None,
                 mask_B: Optional[torch.Tensor] = None):
        super(Phy_SSM, self).__init__()
        self.dim_control = dim_control
        self.encoder = Encoder(input_dim, ssm_input_dim)
        multi_x2z_layers = []
        state_hidden = 128  # s5 x2z layer hidden state dimension
        sample_hidden = 16  # sampled hidden state dimension (recommend to be larger than dim_pc_ssm)
        for _ in range(num_x2z_layers):
            multi_x2z_layers.append(S5Block(ssm_input_dim, state_hidden, bidir=False))
        self.x2z_layers = MultilayerS5Block(multi_x2z_layers)
        self.phy_ssm = Phy_SSM_Unit(input_dim, ode_dim, dim_pc_ssm, sample_hidden, ssm_input_dim, dim_control, n_ssmlayer_A, n_ssmlayer_B, extention,
                                    A_known_const=A_known_const, B_known_const=B_known_const, mask_A=mask_A, mask_B=mask_B)
        # ODE result: z_t to reconstructed input x_t
        self.decoder = Decoder(input_dim, dim_pc_ssm)
        self.prior_mean = nn.Linear(dim_pc_ssm, sample_hidden)
        self.prior_std = nn.Linear(dim_pc_ssm, sample_hidden)
        self.post_mean = nn.Linear(ssm_input_dim, sample_hidden)
        self.post_std = nn.Linear(ssm_input_dim, sample_hidden)

    def compute_time_intervals(self, t):
        """
        Compute time intervals for each sample.
        Pads the first interval with the value of the first actual interval.
        
        Args:
            t: Tensor of shape [batch, seq], time stamps.
        
        Returns:
            intervals: Tensor of shape [batch, seq], where
                    intervals[:, 0] == t[:, 1] - t[:, 0]
                    intervals[:, i] == t[:, i] - t[:, i-1]  for i >= 1
        """
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
        """
        Args:
            mini_batch (Tensor): shape [batch, seq, input_dim], observed state sequence.
            control_data (Tensor or None): shape [batch, seq, dim_control] if used, else None.
            t (Tensor): shape [batch, seq], time stamps for each observation.
        Returns:
            predicted_batch (Tensor): [batch, total_seq, input_dim], reconstructed outputs.
            z_obs (Tensor): [batch, seq, sample_hidden], posterior latent states.
            z_generate (Tensor): [batch, total_seq, sample_hidden], prior latent states.
            (mu_post, sigma_post): posterior distribution.
            (mu_prior, sigma_prior): prior distribution.
        """
        intervals = self.compute_time_intervals(t)
        _, total_seq = intervals.size(0), intervals.size(1)
        _, seq, _ = mini_batch.size(0), mini_batch.size(1), mini_batch.size(2)
        # Encoder
        observation = self.encoder(mini_batch)  # [batch, seq, dim]
        
        z_post = self.x2z_layers(observation, step_scale=intervals[:, :seq])
        latent_z_post_mean = self.post_mean(z_post)
        latent_z_post_std = self.post_std(z_post)
        latent_z = self._reparameterized_sample(latent_z_post_mean, latent_z_post_std)
        #
        batch_size, seq_len, dim_state = observation.shape
        # Phy-SSM
        if self.dim_control is not None:
            control_data = control_data.view(batch_size, total_seq, -1) # (batch, seq , M)
        else:
            control_data = None
        predicted_z, z_obs, z_generate = self.phy_ssm(latent_z, control_data, intervals)
        latent_z_prior_mean = self.prior_mean(z_generate)
        latent_z_prior_std = self.prior_std(z_generate)
        # Decoder
        predicted_batch = self.decoder(predicted_z)
        return predicted_batch, z_obs, z_generate, (latent_z_post_mean, latent_z_post_std), (latent_z_prior_mean, latent_z_prior_std)

if __name__ == "__main__":
    import torch

    # Example forward pass for "Pendulum with unknown friction and unknown control input" model

    # 1) Set batch size and sequence length
    batch_size = 2
    seq_len = 100

    # user-defined parameters for incorporating physical knowledge
    A0 = torch.tensor([[0,1,0,0],
                       [0,0,0,0],
                       [0,0,1,0],
                       [0,0,0,0]], dtype=torch.float32)
    B0 = torch.tensor([[0],[0],[0],[0]], dtype=torch.float32)
    mA = torch.tensor([[0,0,0,0],
                       [1,1,1,1],
                       [0,0,0,0],
                       [0,0,0,0]], dtype=torch.float32)
    mB = torch.tensor([[0],[1],[0],[0]], dtype=torch.float32)

    # 2) Create the model
    model = Phy_SSM(
        input_dim=2,
        ode_dim=2,
        ssm_input_dim=32,
        num_x2z_layers=4,
        dim_pc_ssm=4,  # dim_pc_ssm should equal to ode_dim if no extention, the pendulum example has 2 extend states
        dim_control=1, # if no control input, set it to None
        n_ssmlayer_A=3,
        n_ssmlayer_B=2,
        extention=True,
        A_known_const=A0,
        B_known_const=B0,
        mask_A=mA,
        mask_B=mB,
    )

    # if you don't know about physical knowledge, just drop the parameters, check the example below
    data_driven_SSM = Phy_SSM(
                    input_dim=2,
                    ode_dim=2,
                    ssm_input_dim=32,
                    num_x2z_layers=4,
                    dim_pc_ssm=2,     # dim_pc_ssm should equal to ode_dim if no extention
                    dim_control=None, # if no control input, set it to None
                    n_ssmlayer_A=3,
                    n_ssmlayer_B=2,
                    extention=False
                    )
    model.eval()

    # 3) Create dummy inputs:
    #    mini_batch: [batch, seq_len, 2] for θ(t) and ω(t)
    mini_batch = torch.randn(batch_size, seq_len, 2)

    #    control_data: [batch, seq_len] for torque at each time step
    control_data = torch.randn(batch_size, seq_len, 1)

    #    t: [batch, seq_len] time stamps from 0.0 to 1.0 (can be irregular)
    t = torch.linspace(0.0, 1.0, steps=seq_len).unsqueeze(0).repeat(batch_size, 1)

    # 4) Forward pass
    # input 50 time steps observations to predict 0 to 100 time steps (with control)
    pred, z_obs, z_gen, (mu_post, sigma_post), (mu_prior, sigma_prior) = model(mini_batch[:, :50], control_data, t)
    # input 50 time steps observations to predict 0 to 100 time steps (without control)
    pred, z_obs, z_gen, (mu_post, sigma_post), (mu_prior, sigma_prior) = data_driven_SSM(mini_batch[:, :50], None, t)

    # 5) Print out the shapes of each output tensor
    print("predicted_batch shape:", pred.shape)
    print("z_obs shape:        ", z_obs.shape)
    print("z_gen shape:        ", z_gen.shape)
    print("mu_post shape:      ", mu_post.shape)
    print("sigma_post shape:   ", sigma_post.shape)
    print("mu_prior shape:     ", mu_prior.shape)
    print("sigma_prior shape:  ", sigma_prior.shape)

    # 6) for the loss function, it contains the reconstruction loss, KL divergence and physics state regularization loss
    #    1) reconstruction loss
    recon_loss = F.mse_loss(pred, mini_batch, reduction='mean')
    print("recon_loss:   ", recon_loss.item())
    #    2) KL divergence
    kl_loss = kld_gauss(mu_post, torch.exp(sigma_post), mu_prior, torch.exp(sigma_prior)).mean()
    print("kl_loss:      ", kl_loss.item())
    #    3) physics state regularization loss
    phy_cons_loss = F.mse_loss(z_obs, z_gen, reduction='mean')
    print("phy_cons_loss: ", phy_cons_loss.item())
    #    4) total loss
    beta, gama = 0.1, 1.0 # hyper-parameters
    total_loss = recon_loss + beta * kl_loss + gama * phy_cons_loss
    print("total_loss:   ", total_loss.item())