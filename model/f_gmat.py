
"""### F-GMAT"""

class F_GMAT_base(nn.Module):
    def __init__(self, n_heads, in_channel, num_nodes, dropout, bias=True):
        super(F_GMAT_base, self).__init__()

        self.n_head = n_heads
        self.f_in = num_nodes
        self.a_src = nn.Parameter(torch.Tensor(self.n_head, num_nodes, 1))
        self.a_dst = nn.Parameter(torch.Tensor(self.n_head, num_nodes, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_nodes))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter("bias", None)

        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)

    def forward(self, h):
        bs, ch, n, dim = h.size()
        #h_prime = torch.matmul(h, self.w)
        h_prime = h
        attn_src = torch.matmul(h, self.a_src)
        attn_dst = torch.matmul(h, self.a_dst)
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(
            0, 1, 3, 2
        )
        attn = self.leaky_relu(attn)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime)
        return output + self.bias, attn
        
class F_GMAT(nn.Module):
    def __init__(self, n_heads, in_channel, num_nodes, dropout, alpha):
        super(F_GMAT, self).__init__()
        
        self.dropout = dropout
        
        self.fgmat_layer = F_GMAT_base(
                    n_heads, in_channel, num_nodes, dropout
                )

    def forward(self, x):
        bs,ch,n,dim = x.size()
        x, attn = self.fgmat_layer(x)

        return x


class F_GMAT_module(nn.Module):
    def __init__(self, n_heads, in_channel, num_nodes, mlp, mlp2, dropout, alpha):
        super(F_GMAT_module, self).__init__()
        self.fgmat_net = F_GMAT(n_heads, in_channel, num_nodes, dropout, alpha)

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = 32
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            last_channel = out_channel
        
        self.mlp_convs2 = nn.ModuleList()
        self.mlp_bns2 = nn.ModuleList()
        last_channel = n_heads
        for out_channel in mlp2:
            self.mlp_convs2.append(nn.Conv2d(last_channel, out_channel, 1))
            last_channel = out_channel

        self.lay_norm2 = nn.LayerNorm([n_heads,5, num_nodes])

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
    def forward(self,x, x_f1, x_f2, x_f3, x_f4):
        bs, ch, n, dim = x.size()
      
        x_all = []
        x_f1_all = []
        x_f2_all =[]
        x_f3_all = []
        x_f4_all = []

        # chronological order
        for t_idx in range(1,dim):
            x_input = [x[:,:,:,t_idx].unsqueeze(2),
                    x_f1[:,:,:,t_idx].unsqueeze(2),x_f2[:,:,:,t_idx].unsqueeze(2),
                    x_f3[:,:,:,t_idx].unsqueeze(2),x_f4[:,:,:,t_idx].unsqueeze(2)
                  ]
            x_input = torch.cat(x_input, dim=2)
            x_input_cpy = x_input

            # Encoder
            for i, conv in enumerate(self.mlp_convs):
              x_input = F.relu((conv(x_input)))

            # F-GAMT
            x_input_cpy2 = x_input
            x_input = self.fgmat_net(x_input)
            x_input = x_input_cpy2+ self.dropout1(x_input)

            x_input = self.lay_norm2(x_input)
            
            # Decoder
            for i, conv in enumerate(self.mlp_convs2):
              x_input = F.relu((conv(x_input)))

            x_input = x_input_cpy+ self.dropout2(x_input)

            x_all.append(x_input[:,:,0].unsqueeze(3))
            
        x_tmp = torch.cat(x_all, dim=3)  # (64,16,207,13)
        
        x = torch.cat([x[:,:,:,:1],x_tmp],dim=3)
        
        return x
