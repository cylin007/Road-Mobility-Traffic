


"""### Trainer"""

class Trainer():
    def __init__(self, model, lrate, wdecay, clip, step_size, seq_out_len, scaler, device, cl=True):
        self.scaler = scaler
        self.model = model
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = masked_mae
        self.clip = clip
        self.step = step_size
        self.iter = 1
        self.task_level = 1
        self.seq_out_len = seq_out_len
        self.cl = cl

    def train(self, input, input_f1, input_f2, input_f3, input_f4 ,real_val, idx=None):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input, input_f1, input_f2, input_f3,input_f4, idx=idx)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        
        #for i in range(args.num_nodes):
        #  predict[:,0,i,:] = self.scaler[i].inverse_transform(predict[:,0,i,:])

        if self.iter%self.step==0 and self.task_level<=self.seq_out_len:
            self.task_level +=1
            print("### cl learning\n iter",self.iter,"\niter%step",self.iter%self.step,"\ntask_level",self.task_level)
            print("# predict len:", len(predict[:, :, :, :self.task_level]))
        
        if self.cl:
            loss = masked_mae(predict[:, :, :, :self.task_level], real[:, :, :, :self.task_level], 0.0)
        else:
            loss = masked_mae(predict, real, 0.0)

        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.optimizer.step()
        
        '''
        mae = masked_mae(predict,real,0.0).item()
        mape = masked_mape(predict,real,0.0).item()
        rmse = masked_rmse(predict,real,0.0).item()
        smape = masked_smape(predict,real,0.0).item()
        '''
        metrics = metric(predict, real) # mae,mape,rmse,smape
        
        self.iter += 1
        return metrics # mae,mape,rmse,smape

    def eval(self, input, input_f1, input_f2, input_f3,input_f4, real_val):
        self.model.eval()
        output = self.model(input, input_f1, input_f2, input_f3,input_f4)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)
        predict = self.scaler.inverse_transform(output)
        
        '''
        loss = self.loss(predict, real, 0.0)
        mape = masked_mape(predict,real,0.0).item()
        rmse = masked_rmse(predict,real,0.0).item()
        '''    
        metrics = metric(predict, real) # mae,mape,rmse,smape
        return metrics # mae,mape,rmse,smape

"""### Parameter"""

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')