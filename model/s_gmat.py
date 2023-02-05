
"""### S-GMAT"""

"""### Model"""

def pearson_corr2(all, num_nodes): # all: (64,8,n,dim)
  #all = torch.randn(64,8,10,15)
  
  all_cor = []
  for i in range(num_nodes):
    a = all[:,i].unsqueeze(1)
    b = all.clone()
    
    corrs = (a*b).mean(axis=-1) - a.mean(axis=-1) * b.mean(axis=-1)
    #print(corrs)
    std_prod =  torch.sqrt(torch.var(a, dim=-1) * torch.var(b, dim=-1))

    out = corrs / std_prod

    out = torch.where(torch.isnan(out), 0, out)
    
    all_cor.append((out).unsqueeze(1))

  all_cor = torch.cat(all_cor, dim=1)
  all_cor = torch.sigmoid(all_cor)
  return all_cor

class S_GMAT_base(nn.Module):
    def __init__(self, n_heads, in_channel, num_nodes, dropout, bias=True):
        super(S_GMAT_base, self).__init__()

        self.n_head = n_heads
        self.f_in = num_nodes
        self.a_src = nn.Parameter(torch.Tensor(self.n_head, num_nodes, 1))
        self.a_dst = nn.Parameter(torch.Tensor(self.n_head, num_nodes, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)
        
    def forward(self, h, adj, s_attn):

        bs, ch, n, dim = h.size()
        
        attn_src = torch.matmul(h, self.a_src)
        attn_dst = torch.matmul(h, self.a_dst)
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(
            0, 1, 3, 2
        )
        attn = self.leaky_relu(attn)
        
        zero_vec = -9e15*torch.ones_like(attn)
        attn = torch.where(adj > 0, attn, zero_vec) 
        attn = self.softmax(attn) # bs x n_head x n x n

        attn = attn + s_attn

        alpha = 0.05
        all = [h]
        h_prime = h
        h_prime = alpha*h+ (1-alpha)* torch.matmul(attn, h_prime)
        all.append(h_prime)
        h_prime = alpha*h+ (1-alpha)* torch.matmul(attn, h_prime)
        all.append(h_prime)

        return torch.cat(all, dim=1)
class S_GMAT(nn.Module):
    def __init__(self, n_heads, in_channel, num_nodes, dropout, alpha):
        super(S_GMAT, self).__init__()
        
        self.dropout = dropout
        
        self.sgmat_layer = S_GMAT_base(
                    n_heads, in_channel, num_nodes, dropout
                )

    def forward(self, x, adj, s_attn):
        bs,ch,n,dim = x.size()

        x = self.sgmat_layer(x, adj, s_attn)

        return x


class S_GMAT_module(nn.Module):
    def __init__(self, depth, temporal_len, n_heads, in_channel, num_nodes, mlp, mlp2, dropout, alpha):
        super(S_GMAT_module, self).__init__()
        

        self.sgmat_net = S_GMAT(n_heads, in_channel, temporal_len, dropout, alpha)
        
        self.mlp_convs_start_1 = nn.Conv2d(in_channel, n_heads, 1)
        self.mlp_convs_start_2 = nn.Conv2d(in_channel, n_heads, 1)

        self.mlp_convs_end_1 = nn.Conv2d(n_heads*(1+depth), 32, 1)
        self.mlp_convs_end_2 = nn.Conv2d(n_heads*(1+depth), 32, 1)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

        self.m1 = nn.GroupNorm((1+depth), n_heads*(1+depth))
        self.m2 = nn.GroupNorm((1+depth), n_heads*(1+depth))

        self.mlp1 = (nn.Conv2d(32,32,(1,1))) 
        self.mlp2 = (nn.Conv2d(32,32,(1,1))) 

        self.norm1 = nn.LayerNorm([32, num_nodes, temporal_len])
        self.norm2 = nn.LayerNorm([32, num_nodes, temporal_len])

    def forward(self,x,adj1,adj2, pearson_attn):
        
        bs, ch, n, dim = x.size()
        
        x_input = x.clone()
        #Encoder
        x_input = self.mlp_convs_start_1(x_input)

        # S-GMAT
        x_input = F.elu(self.sgmat_net(x_input,adj1, pearson_attn))

        #Decoder
        x_input = self.mlp_convs_end_1(x_input)
        
        x_input = (x + self.dropout1(x_input))

        x_input = F.elu(self.mlp1(x_input))
        x_input1 = self.norm1(x_input)

        return x_input
