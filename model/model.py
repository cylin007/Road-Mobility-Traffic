
class GMAT_Net(nn.Module):
    def __init__(self, 
                 model_type, 
                 num_nodes, 
                 device, 
                 predefined_A=None,
                 dropout=0.3, 
                 dilation_exponential=1, 
                 conv_channels=32, 
                 residual_channels=32, 
                 skip_channels=64, 
                 end_channels=128, 
                 seq_length=12, in_dim=2, out_dim=12, layers=3, layer_norm_affline=True):
      
        super(GMAT_Net, self).__init__()

        self.model_type = model_type

        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.layers = layers
        self.seq_length = seq_length

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.s_gmat = nn.ModuleList()

        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.f_gmat = nn.ModuleList()
        in_channel = 32
        n_heads = 8
        dropout = 0
        alpha = 0.2
        self.f_gmat.append(
            F_GMAT_module(
              n_heads=n_heads, in_channel= in_channel, num_nodes=num_nodes, mlp=[n_heads],mlp2=[32], dropout=dropout, alpha=alpha
            )
        )

        self.t_gmat1 = nn.ModuleList()
        self.t_gmat2 = nn.ModuleList()

        self.receptive_field = 13
        print("# Model Type", self.model_type)
        print("# receptive_field", self.receptive_field)
        i=0

        target_len = 13
        for j in range(1,layers+1):
           
            kern = 5

            dilation_factor = 1
            
            in_channel = 32
            n_heads = 8
            dropout = 0
            alpha = 0.2
            self.t_gmat1.append(
                T_GMAT_module(
                  kern= kern, dilation_factor=dilation_factor, temporal_len = target_len, n_heads=n_heads, in_channel= in_channel, num_nodes=num_nodes, mlp=[n_heads],mlp2=[32], dropout=dropout, alpha=alpha
                )
            )
            
            self.t_gmat2.append(
                T_GMAT_module(
                  kern= kern, dilation_factor=dilation_factor, temporal_len = target_len, n_heads=n_heads, in_channel= in_channel, num_nodes=num_nodes, mlp=[n_heads],mlp2=[32], dropout=dropout, alpha=alpha
                )
            )
            target_len -= 4

            self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                            out_channels=skip_channels,
                                            kernel_size=(1, target_len)))

            in_channel = 32
            n_heads = 8
            dropout = 0
            alpha = 0.2
            
            depth = 2
            self.s_gmat.append(
                S_GMAT_module(
                  depth=depth, temporal_len = target_len, n_heads=n_heads, in_channel= in_channel, num_nodes=num_nodes, mlp=[n_heads],mlp2=[32], dropout=dropout, alpha=alpha
                )
            )

            self.norm.append(LayerNorm((residual_channels, num_nodes, target_len),elementwise_affine=layer_norm_affline))
        
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                             out_channels=end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                             out_channels=out_dim,
                                             kernel_size=(1,1),
                                             bias=True)
        
        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)

        self.idx = torch.arange(self.num_nodes).to(device)

    def forward(self, input, input_f1,input_f2,input_f3,input_f4, idx=None):
        seq_len = input.size(3)
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'

        bs,c,n,dim= input.shape

        # Pearson Correlation
        pearson_attn = pearson_corr2(input[:,0], n).unsqueeze(1) #input: 64,n,dim, attn: 64,n,n

        # Step0: 檢查receptive_field, 不足則padding0
        if self.seq_length<self.receptive_field:
            input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))
            input_f1 = nn.functional.pad(input_f1,(self.receptive_field-self.seq_length,0,0,0))
            input_f2 = nn.functional.pad(input_f2,(self.receptive_field-self.seq_length,0,0,0))
            input_f3 = nn.functional.pad(input_f3,(self.receptive_field-self.seq_length,0,0,0))
            input_f4 = nn.functional.pad(input_f4,(self.receptive_field-self.seq_length,0,0,0))

        # Step1: turn([64, 2, 207, 13]) to ([64, 32, 207, 13]) => 固定用同一conv
        x = self.start_conv(input) 
        x_f1 = self.start_conv(input_f1)  
        x_f2 = self.start_conv(input_f2)
        x_f3 = self.start_conv(input_f3)  
        x_f4 = self.start_conv(input_f4) 

        x = self.f_gmat[0](x,x_f1,x_f2,x_f3,x_f4)

        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        
        # Layers : 3層 : 19->13->7->1 (取決於TCN取的維度)
        for i in range(self.layers):
            
            residual = x    
            
            ### T-GMAT -START###
            filter = self.t_gmat1[i](x)
            filter = torch.tanh(filter)

            gate = self.t_gmat2[i](x)
            gate = torch.sigmoid(gate)
            ### T-GMAT -END###

            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)

            s = x
            s = self.skip_convs[i](s)    
            skip = s + skip
            
            ### S-GMAT -START###
            x = self.s_gmat[i](x, self.predefined_A[0], self.predefined_A[1], pearson_attn)
            ### S-GMAT -START###

            x = x + residual[:, :, :, -x.size(3):]
            
            # Based on MTGNN: https://github.com/nnzhan/MTGNN
            if idx is None:
                x = self.norm[i](x,self.idx)
            else:
                x = self.norm[i](x,idx)

        skip = self.skipE(x) + skip

        # Based on MTGNN
        ### Output Module -START### 
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        ### Output Module -START###

        return x