import torch.nn as nn
import torch
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
# Use einsum to do batched matrix-vector multiplication
        # 'bjk,bk->bj' is the Einstein summation convention
        # b: batch size, j/k: state dimension
        # dA_known_matrix = torch.einsum('bjk,bk->bj', A_known_matrix, input_x)
class SSM_step_ODE(nn.Module):
    def __init__(self, dim_control):
        '''Calculate the next state according to the dz & u'''
        super(SSM_step_ODE, self).__init__()
        self.dim_control = dim_control
        
    def forward(self, Unk_A, Unk_B, z_input, u_input):
        ''' Return the continuous system dynamics A, B.'''
        if Unk_A.dim() == 3:  # Shape: [batch, seq, N]
            return self.get_seq_dynamics(Unk_A, Unk_B, z_input, u_input)
        else:
            return self.get_dynamics(Unk_A, Unk_B, z_input, u_input)
    
    def get_dynamics(self, Unk_A, Unk_B, z_input, u_input):
        '''
        Return the continuous system dynamics A, B.
        Unk_A.shape = [batch, 16]
        Unk_B.shape = [batch, 4]
        '''
        input_x = z_input
        batch_size, dim_state = input_x.shape[0], input_x.shape[1]
        '''Get A_known here'''
        # Create the constant part of matrix A
        A_known_const = torch.tensor([[0, 0, 0],
                                      [0, 1, 1],
                                      [0, 1, 0]], dtype=torch.float32)
        # Initialize the dynamic part of matrix A to zero
        A_known_dynamic = torch.zeros((batch_size, dim_state, dim_state), dtype=torch.float32)
        
        # Update the dynamic part of matrix A with omega(t) values
        A_known_dynamic[:, 0, 0] = -input_x[:, 1]  # Set -x2 at the right position
        A_known_dynamic[:, 1, 0] = input_x[:, 1]  # Set x2 at the right position
        
        # Combine the constant and dynamic parts
        A_known_matrix = A_known_const[None, :, :] + A_known_dynamic  # Add batch dimension to A_const
        A_known_matrix = A_known_matrix.to(input_x.device)
        '''Get A_unknown here'''
        # Known mask matrix representing positions of unknown dynamics
        Mask_A = torch.tensor([[1, 0, 0],
                               [1, 1, 1],
                               [0, 1, 0]], dtype=torch.float32, device=input_x.device)

        # Expand the mask to match the batch size
        Mask_A = Mask_A.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Reshape Unk_A and apply Mask_A to get A_unknown_matrix
        A_unknown_matrix = Unk_A.view(batch_size, dim_state, dim_state) * Mask_A
        '''Get B_unknown here'''
        if self.dim_control is not None:
            # Known mask matrix representing positions of unknown dynamics for B
            Mask_B = torch.tensor([[0], [1], [0], [0]], dtype=torch.float32, device=input_x.device)
            
            # Expand the mask to match the batch size
            Mask_B = Mask_B.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Reshape Unk_B and apply Mask_B to get B_unknown_matrix
            B_unknown_matrix = Unk_B.view(batch_size, dim_state, self.dim_control) * Mask_B
        else:
            B_unknown_matrix = None
        
        # Get system dynamics matrices
        A_matrix = A_known_matrix * A_unknown_matrix
        B_matrix = B_unknown_matrix
        
        return A_matrix, B_matrix
    def get_seq_dynamics(self, Unk_A, Unk_B, z_input, u_input):
        '''
        Return the continuous system dynamics A, B.
        Unk_A.shape = [batch, seq, dim]
        Unk_B.shape = [batch, seq, dim] or [batch, dim] if no sequence
        z_input.shape = [batch, seq, dim]
        '''
        batch_size, seq_len, dim_state = z_input.shape[0], z_input.shape[1], z_input.shape[2]

        # Create the constant part of matrix A
        A_known_const = torch.tensor([[0, 0, 0],
                                      [0, 1, 1],
                                      [0, 1, 0]], dtype=torch.float32)

        # Initialize the dynamic part of matrix A to zero
        A_known_dynamic = torch.zeros((batch_size, seq_len, dim_state, dim_state), dtype=torch.float32)
        
        # Update the dynamic part of matrix A with omega(t) values
        A_known_dynamic[:, :, 0, 0] = -z_input[:, :, 1]  # Set -x2 at the right position
        A_known_dynamic[:, :, 1, 0] = z_input[:, :, 1]  # Set x2 at the right position
        
        # Combine the constant and dynamic parts
        A_known_matrix = A_known_const[None, None, :, :] + A_known_dynamic  # Add batch and seq dimensions
        A_known_matrix = A_known_matrix.to(z_input.device)

        # Known mask matrix representing positions of unknown dynamics
        Mask_A = torch.tensor([[1, 0, 0],
                               [1, 1, 1],
                               [0, 1, 0]], dtype=torch.float32, device=z_input.device)

        # Expand the mask to match the batch and sequence length
        Mask_A = Mask_A.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)
        
        # Reshape Unk_A and apply Mask_A to get A_unknown_matrix
        A_unknown_matrix = Unk_A.view(batch_size, seq_len, dim_state, dim_state) * Mask_A

        if self.dim_control is not None:
            # Known mask matrix representing positions of unknown dynamics for B
            Mask_B = torch.tensor([[0], [1], [0], [0]], dtype=torch.float32, device=z_input.device)
            
            # Expand the mask to match the batch and sequence length
            Mask_B = Mask_B.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)
            
            # Reshape Unk_B and apply Mask_B to get B_unknown_matrix
            B_unknown_matrix = Unk_B.view(batch_size, seq_len, dim_state, self.dim_control) * Mask_B
        else:
            B_unknown_matrix = None
        
        # Get system dynamics matrices
        A_matrix = A_known_matrix * A_unknown_matrix 
        B_matrix = B_unknown_matrix
        
        return A_matrix, B_matrix

class Physics_constraint_SSM(nn.Module):
    def __init__(self, dim_state=64, dim_control=None, n_ssmlayer_A=2, n_ssmlayer_B=2):
        super(Physics_constraint_SSM, self).__init__()
        self.dim_state = dim_state  # Number of original state dimensions of (u,x)
        self.dim_control = dim_control  # Number of control signal dimensions
        state_hidden = 128  # s5 state dimension
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
                                      nn.Linear(state_hidden, 9))
        self.SSM_Matrix = SSM_step_ODE(dim_control)

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
    
    def forward(self, interval, input_x, input_ux, **kwargs):
        """
        t: time; shape = torch.rand((seq_len))
        input_x: the system state; shape = torch.rand((batch_size, seq_len, N)) 
        input_u (optional): the control signal; shape = torch.rand((batch_size, seq_len, M)) 
        return the system's state; torch.rand((batch_size, seq_len, N)) 
        """

        batch_size, seq_len, dim_state = input_x.shape
        Unk_A, next_state_z = self.ssm_A_layers(input_ux, step_scale=interval, return_state=True)
        # Approximate the unknown system dynamics function by SSM
        Unk_A = self.ssm_to_A(Unk_A)
        if self.dim_control is not None:
            Unk_B = Unk_A
        else:
            Unk_B = Unk_A
            control_data = Unk_B
        A, B = self.SSM_Matrix(Unk_A, Unk_B, input_x, control_data)
        # Discretize A and B matrices using bilinear transformation
        dA, dB = self.bilinear(interval, A, B)
        # Update state
        next_pred = self.update_state(dA, dB, input_x, control_data)
        return next_pred, next_state_z
    
    def step(self, interval, input_x, input_ux, state_z, **kwargs):
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
            Unk_B = Unk_A
        else:
            Unk_B = Unk_A
            control_data = Unk_B

        A, B = self.SSM_Matrix(Unk_A, Unk_B, input_x, control_data)
        # Discretize A and B matrices using bilinear transformation
        dA, dB = self.bilinear(interval, A, B)
        # Update state
        next_pred = self.update_state(dA, dB, input_x, control_data)
        return next_pred, next_state_z

class Encoder(nn.Module):
    def __init__(self, input_dim, ode_dim, ssm_input_dim, ssm_dropout):
        '''
        Attention: ode_dim is the dim of original ode state, not the extention form. For instance, ode_dim = 2 for pendulum 
        '''
        super(Encoder, self).__init__()
        self.first_layer = nn.Linear(input_dim, 200)
        self.second_layer = nn.Linear(200, 200)
        self.third_layer = nn.Linear(200, ssm_input_dim)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
    def forward(self, input_batch, interval):
        # Create data batch for the SSM
        out = input_batch
        out = self.sigmoid(self.first_layer(out))
        out = out + self.sigmoid(self.second_layer(out))
        out = self.sigmoid(self.third_layer(out))
        batch_size, sequence_length, _ = out.size()
        return out   # observation

class Decoder(nn.Module):
    def __init__(self, input_dim, dim_pc_ssm, ssm_input_dim, dim_control=None, ssm_dropout=0.1, n_ssmlayer_A = 2, 
                 n_ssmlayer_B=2, extention=True):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.extention = extention  # Whether to extend the system state
        self.dim_control = dim_control
        # Control Signal lift
        if self.dim_control is not None:
            self.control_to_latent = nn.Sequential(nn.Linear(dim_control, ssm_input_dim), nn.LeakyReLU())
        else:
            self.control_to_latent = nn.Identity()
        # Physics_constraint_SSM
        self.Physics_constraint_SSM = Physics_constraint_SSM(dim_state=3, dim_control=dim_control,
                                                             n_ssmlayer_A = n_ssmlayer_A, n_ssmlayer_B=n_ssmlayer_B)
        self.dim_pc_ssm = dim_pc_ssm 
        
        # ODE result: z_t to reconstructed input x_t
        self.first_layer = nn.Linear(dim_pc_ssm, 200)
        self.second_layer = nn.Linear(200, input_dim)

        # Activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.latent2z = nn.Linear(16, 3) # 3 is ode dim

    def forward(self, latent_z_batch, control_data, interval):
        batch_size, seq_len, dim_state = latent_z_batch.shape[0], latent_z_batch.shape[1], latent_z_batch.shape[2]    # Note that this "batch" = num_particles * batch
        _, total_len = interval.shape[0], interval.shape[1]
        device = latent_z_batch.device
        # Lift the control data & z0
        latent_control = self.control_to_latent(control_data)
        z = self.latent2z(latent_z_batch)
        # get the (u,z) latent as SSM input
        if self.dim_control is not None:
            latent_uz = z + latent_control[:, :seq_len, :] # first step (u,z)
        else:
            latent_uz = z
        z_state, state_z_ssm = self.Physics_constraint_SSM(interval[:, :seq_len], z, latent_uz)
        z_obs = z
        z_generate = torch.cat([z[:, 0, :].unsqueeze(1), z_state[:, :-1, :]], dim=1)
        # 
        n_steps = total_len - seq_len
        z_extra = z_state[:, -1, :]
        z_extra_seq = torch.empty(batch_size, n_steps, z_extra.shape[1], device=device)
        for i in range(total_len - seq_len):    # t_1 to xxx
            z_extra, state_z_ssm = self.Physics_constraint_SSM.step(interval[:, i + seq_len], z_extra, z_extra, state_z_ssm)
            z_extra_seq[:, i, :] = z_extra
        z_final = torch.cat([z_state, z_extra_seq], dim=1)
        recon_batch = self.sigmoid(self.first_layer(z_final))
        recon_batch = self.second_layer(recon_batch)
        return recon_batch, z_obs, z_generate


################################################################################################
# discrete continuous ssm with bilinear method to get the next system state prediction
################################################################################################
if __name__ == '__main__':
    import torch

    def bilinear(dt, A, B=None):
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

    def update_state(dA, dB, X, U):
        if dA.dim() == 4:  # Shape: [batch, seq, N, N]
            X_next = torch.einsum('bsij,bsj->bsi', dA, X) + torch.einsum('bsij,bsj->bsi', dB, U)
        elif dA.dim() == 3:  # Shape: [batch, N, N]
            X_next = torch.einsum('bij,bj->bi', dA, X) + torch.einsum('bij,bj->bi', dB, U)
        return X_next

    # Example 1: Sequential data
    batch_size, seq_len, N, M = 2, 3, 2, 4
    torch.manual_seed(0)
    dt_seq = torch.rand((seq_len))
    A_seq = torch.rand((batch_size, seq_len, N, N))
    B_seq = torch.rand((batch_size, seq_len, N, M))
    X_seq = torch.rand((batch_size, seq_len, N))
    U_seq = torch.rand((batch_size, seq_len, M))

    dA_seq, dB_seq = bilinear(dt_seq, A_seq, B_seq)
    X_next_seq = update_state(dA_seq, dB_seq, X_seq, U_seq)

    print("Next state with sequence (X_next_seq):", X_next_seq)

    # Example 2: Single step data
    batch_size, N, M = 2, 2, 4
    torch.manual_seed(0)
    dt_single = torch.rand((1))
    A_single = torch.rand((batch_size, N, N))
    B_single = torch.rand((batch_size, N, M))
    X_single = torch.rand((batch_size, N))
    U_single = torch.rand((batch_size, M))

    dA_single, dB_single = bilinear(dt_single, A_single, B_single)
    X_next_single = update_state(dA_single, dB_single, X_single, U_single)

    print("Next state with single step (X_next_single):", X_next_single)
