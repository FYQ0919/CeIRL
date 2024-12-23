import torch.nn as nn
import numpy as np
import torch
from torch.distributions import Normal
from utils.torch_utils import  get_activation,init_orhtogonal,SquashedGaussian
from .adaptation_module import body_index,hip_index,thigh_index,calf_index,edge_index,GraphEncoder,mlp 
from torch_geometric.nn import ResGatedGraphConv, GatedGraphConv

class GraphForward(nn.Module):
    def __init__(self, 
                 num_obs,
                 num_latent,
                 num_actions,
                 activation = 'elu',):
        super().__init__()
        """
        obs + action + latent -> next_obs 
        obs -> node 
        action -> node
        """
        self.num_latent = num_latent
        # graph info
        node_base = torch.tensor(list(body_index.values()),dtype=torch.long).squeeze()
        node_hip = torch.stack([torch.tensor(list(hip_index.values()),dtype=torch.long)],dim=0).squeeze()
        node_thigh = torch.stack([torch.tensor(list(thigh_index.values()),dtype=torch.long)],dim=0).squeeze()
        node_calf = torch.stack([torch.tensor(list(calf_index.values()),dtype=torch.long)],dim=0).squeeze()
        
        node_hip_action = torch.tensor([0, 3, 6, 9],dtype=torch.long)
        node_thigh_action = torch.tensor([1, 4, 7, 10],dtype=torch.long)
        node_calf_action = torch.tensor([2, 5, 8, 11],dtype=torch.long)
        self.node_hip_action = nn.Parameter(node_hip_action, requires_grad=False)
        self.node_thigh_action = nn.Parameter(node_thigh_action, requires_grad=False)
        self.node_calf_action = nn.Parameter(node_calf_action, requires_grad=False)

        activation_fn = get_activation(activation)
        self.node_base = nn.Parameter(node_base, requires_grad=False)
        self.node_hip = nn.Parameter(node_hip, requires_grad=False)
        self.node_thigh = nn.Parameter(node_thigh, requires_grad=False)
        self.node_calf = nn.Parameter(node_calf, requires_grad=False) 
        self.edge = nn.Parameter(torch.as_tensor(edge_index, dtype=torch.long).contiguous().t(),requires_grad=False)
        # pipeline 
        base_input_size =  len(list(body_index.values())[0])
        hip_input_size = len(list(hip_index.values())[0]) + 1
        thigh_input_size =len(list(thigh_index.values())[0]) + 1 
        calf_input_size = len(list(calf_index.values())[0]) + 1 
        self.base_net = mlp(base_input_size, num_latent, [128], activation_fn)
        self.hip_net = mlp(hip_input_size, num_latent, [128], activation_fn)
        self.thigh_net = mlp(thigh_input_size, num_latent, [128], activation_fn)
        self.calf_net = mlp(calf_input_size, num_latent, [128], activation_fn)
        # graph neural network 
        self.gn = ResGatedGraphConv(in_channels=2 * num_latent,
                                    out_channels=2 * num_latent)
        self.act = activation_fn
        self.gn2 = ResGatedGraphConv(in_channels=2 * num_latent,
                                    out_channels=num_latent)
        # decoder 
        ## base_vel, base_ang, project_gravity, cmd
        ## dof_pos,vel,actions, contact
        self.base_decoder = mlp(num_latent, 3 + 3 + 3 + 3, [128], activation_fn)
        self.leg_decoder = mlp(num_latent * 4 , 3 + 3 + 3 + 1, [128], activation_fn)
        self.FL_Leg = nn.Parameter(torch.tensor([0,1,5,9],dtype=torch.long), requires_grad=False)
        self.FR_Leg = nn.Parameter(torch.tensor([0,2,6,10],dtype=torch.long), requires_grad=False)
        self.RL_Leg = nn.Parameter(torch.tensor([0,3,7,11],dtype=torch.long), requires_grad=False)
        self.RR_Leg = nn.Parameter(torch.tensor([0,4,8,12],dtype=torch.long), requires_grad=False)


    def _obsaction2node(self,obs,action):
        base = obs[:,self.node_base].unsqueeze(1) # (bz, 1, n_base)
        hip = obs[:,self.node_hip]# (bz, 4, 4)
        thigh = obs[:,self.node_thigh]
        calf = obs[:,self.node_calf]
        hip_action = action[:,self.node_hip_action].unsqueeze(-1)
        thigh_action = action[:,self.node_thigh_action].unsqueeze(-1)
        calf_action = action[:,self.node_calf_action].unsqueeze(-1)
        base = self.base_net(base)
        hip = self.hip_net(torch.cat([hip,hip_action],dim=-1))
        thigh = self.hip_net(torch.cat([thigh,thigh_action],dim=-1))
        calf = self.calf_net(torch.cat([calf,calf_action],dim=-1))
        node = torch.cat([base,hip,thigh,calf],dim=1)
        return node # shape (bz, 13, 4)

    def forward(self,obs,action,latent):
        obsaction_node = self._obsaction2node(obs,action) 
        nodes_latent = torch.cat([obsaction_node,latent],dim=-1) # (bz, n_node, 2*num_latent)
        nodes_latent = self.gn(nodes_latent, self.edge) # (bz, n_node, 2*num_latent)
        nodes_latent = self.act(nodes_latent)
        nodes_latent = self.gn2(nodes_latent, self.edge) # (bz, n_node, num_latent) 

        Base_latent = nodes_latent[:,0:1,:].reshape(-1,self.num_latent)
        FL_Leg_latent = nodes_latent[:,self.FL_Leg,:].reshape(-1,4*self.num_latent)
        FR_Leg_latent = nodes_latent[:,self.FR_Leg,:].reshape(-1,4*self.num_latent)
        RL_Leg_latent = nodes_latent[:,self.RL_Leg,:].reshape(-1,4*self.num_latent)
        RR_Leg_latent = nodes_latent[:,self.RR_Leg,:].reshape(-1,4*self.num_latent)

        base_decoded = self.base_decoder(Base_latent) # (bz,12)
        FL_Leg_decoded = self.leg_decoder(FL_Leg_latent) # (bz, 4) 
        FR_Leg_decoded = self.leg_decoder(FR_Leg_latent)
        RL_Leg_decoded = self.leg_decoder(RL_Leg_latent)
        RR_Leg_decoded = self.leg_decoder(RR_Leg_latent)
        decoded_pos = torch.cat([FL_Leg_decoded[:,0:3],FR_Leg_decoded[:,0:3],RL_Leg_decoded[:,0:3],RR_Leg_decoded[:,0:3]],dim=-1) # (bz,12)
        decoded_vel = torch.cat([FL_Leg_decoded[:,3:6],FR_Leg_decoded[:,3:6],RL_Leg_decoded[:,3:6],RR_Leg_decoded[:,3:6]],dim=-1)
        decoded_act = torch.cat([FL_Leg_decoded[:,6:9],FR_Leg_decoded[:,6:9],RL_Leg_decoded[:,6:9],RR_Leg_decoded[:,6:9]],dim=-1)
        decoded_contact = torch.cat([FL_Leg_decoded[:,9:10],FR_Leg_decoded[:,9:10],RL_Leg_decoded[:,9:10],RR_Leg_decoded[:,9:10]],dim=-1)
        decoded = torch.cat((base_decoded,decoded_pos,decoded_vel,decoded_act,decoded_contact),dim=-1)
        return decoded

class GRUAdaptationNetwork(nn.Module):
    def __init__(self, input_size, seq_len, dim, hidden_size, act_fn):
        super(GRUAdaptationNetwork, self).__init__()
        self.seq_len = seq_len
        self.dim = dim
        # 将输入的特征维度转换为GRU的输入维度
        self.pre_gru = nn.Linear(input_size, dim)
        # GRU层
        self.gru = nn.GRU(input_size=dim, hidden_size=hidden_size*2, batch_first=True)
        # 在GRU之后，我们可以有一个全连接层来调整输出维度（如果需要的话）
        self.post_gru = nn.Linear(hidden_size*2, hidden_size)

        # 激活函数
        self.act_fn = act_fn

    def forward(self, x):
        # 假设x的原始形状是(batch_size, input_size)
        # 我们首先通过一个线性层来调整维度以匹配GRU的期望输入
        # print(x.shape)
        # x = self.pre_gru(x)
        # print(x.shape)
        # 然后我们需要改变x的形状以符合GRU的输入要求：(batch, seq_len, feature)
        x1 = x.view(-1, self.seq_len, self.dim)
        # 通过GRU层
        x2, _ = self.gru(x1)
        # 我们只取序列的最后一个元素来进行下一步处理
        x2 = x2[:, -1, :]
        # 通过后续处理层
        x3 = self.post_gru(x2)
        x4 = self.act_fn(x3)
        return x4

class MLPForward(nn.Module):
    def __init__(self, 
                 num_obs,
                 num_latent,
                 num_actions,
                 activation = 'elu',):
        super().__init__()
        self.num_latent = num_latent
        # mlp_input_size = num_obs + num_actions + num_latent
        activation_fn = get_activation(activation)
        self.mlp = mlp(self.num_latent, num_obs, [512,256,128], activation_fn)
        self.leg_mask = mlp(self.num_latent, 12, [256,128,64], activation_fn)

    def forward(self, latent):
        decoded = self.mlp(latent)
        leg_mask_logits = self.leg_mask(latent)
        leg_mask = torch.sigmoid(leg_mask_logits)  # Apply sigmoid activation
        leg_mask_binary = (leg_mask > 0.5).float()  # Apply thresholding to get binary values (0 or 1)
        # print(leg_mask_binary)
        return decoded, leg_mask_binary
        # return decoded

class MLPFFT(nn.Module):
    def __init__(self,
                 num_obs,
                 num_latent,
                 num_actions,
                 activation = 'elu',):
        super().__init__()
        self.num_latent = num_latent
        # mlp_input_size = num_obs + num_actions + num_latent
        activation_fn = get_activation(activation)
        self.mlp = mlp(num_obs, num_latent, [512,256,128], activation_fn)


    def forward(self, latent):
        decoded = self.mlp(latent)

        return decoded
