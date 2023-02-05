



"""### Training Model (GMAT-Net)"""

def main(runid):
    
    model = GMAT_Net(
        args.model_type, 
        args.num_nodes,
        device, 
        predefined_A=adj_mx, 
        dropout=args.dropout, 
        conv_channels=args.conv_channels, 
        residual_channels=args.residual_channels,
        skip_channels=args.skip_channels, 
        end_channels= args.end_channels,
        seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,  layers=args.layers, layer_norm_affline=True)


    nParams = sum([p.nelement() for p in model.parameters()])       # model參數量!
    print('Number of model parameters is', nParams)

    engine = Trainer(model, args.learning_rate, args.weight_decay, args.clip, args.step_size1, args.seq_out_len, data['scaler'], device, args.cl)
    
    print("start training...",flush=True)
    his_loss =[]
    val_time = []
    train_time = []
    minl = 1e5
    start_epoch=0
    train_loss_epoch = []  # 紀錄train在epoch收斂
    valid_loss_epoch = []  # 紀錄valid在epoch收斂
    
    '''
    #####
    SAVE_PATH = args.save + "exp202112290828_0.pth" 
    checkpoint = torch.load(SAVE_PATH)
    engine.model.load_state_dict(checkpoint['model_state_dict'])
    engine.task_level = checkpoint['task_level']
    start_epoch = checkpoint['epoch']
    train_loss_epoch = checkpoint['train_loss']
    valid_loss_epoch = checkpoint['valid_loss']
    #####
    '''
    
    for i in range(start_epoch,start_epoch+args.epochs+1):
        train_mae = []
        train_mape = []
        train_smape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()  # 為了檢視資料先拿掉
        for iter, (x, y,x_f1,x_f2,x_f3,x_f4) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx= trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            
            trainx_f1 = torch.Tensor(x_f1).to(device)
            trainx_f1= trainx_f1.transpose(1, 3)
            
            trainx_f2 = torch.Tensor(x_f2).to(device)
            trainx_f2= trainx_f2.transpose(1, 3)
            
            
            trainx_f3 = torch.Tensor(x_f3).to(device)
            trainx_f3= trainx_f3.transpose(1, 3)
            
            trainx_f4 = torch.Tensor(x_f4).to(device)
            trainx_f4= trainx_f4.transpose(1, 3)
            
            #mae,mape,rmse,smape
            metrics = engine.train(trainx,trainx_f1,trainx_f2,trainx_f3,trainx_f4 ,trainy[:,0,:,:])

            train_mae.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            train_smape.append(metrics[3])

            if iter % args.print_every == 0 :
                log = 'Iter: {:03d}, Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, *Train SMAPE: {:.4f}'
                print(log.format(iter, train_mae[-1], train_mape[-1], train_rmse[-1], train_smape[-1]),flush=True)
        t2 = time.time()
        train_time.append(t2-t1)
        #validation
        valid_mae = []
        valid_mape = []
        valid_rmse = []
        valid_smape = []

        s1 = time.time()
        for iter, (x, y,x_f1,x_f2,x_f3,x_f4)  in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            
            testx_f1 = torch.Tensor(x_f1).to(device)
            testx_f1= testx_f1.transpose(1, 3)
            
            testx_f2 = torch.Tensor(x_f2).to(device)
            testx_f2= testx_f2.transpose(1, 3)
            
            
            testx_f3 = torch.Tensor(x_f3).to(device)
            testx_f3= testx_f3.transpose(1, 3)
            
            testx_f4 = torch.Tensor(x_f4).to(device)
            testx_f4= testx_f4.transpose(1, 3)
            
            
            metrics = engine.eval(testx, testx_f1,testx_f2,testx_f3,testx_f4, testy[:,0,:,:])
            valid_mae.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
            valid_smape.append(metrics[3])
            
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_mae = np.mean(train_mae)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)
        mtrain_smape = np.mean(train_smape)

        mvalid_mae = np.mean(valid_mae)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        mvalid_smape = np.mean(valid_smape)
        
        #his_loss.append(mvalid_loss)
        his_loss.append(mvalid_smape)

        log = 'Epoch: {:03d}, Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, *Train SMAPE: {:.4f}, Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, *Valid SMAPE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_mae, mtrain_mape, mtrain_rmse, mtrain_smape, mvalid_mae, mvalid_mape, mvalid_rmse, mvalid_smape, (t2 - t1)),flush=True)
        
        train_loss_epoch.append(mtrain_mae)
        valid_loss_epoch.append(mvalid_mae)
        
        if mvalid_mae<minl:
            target_best_model = args.save + "exp" + str(args.expid) + "_" + str(runid) +".pth"
            print("### Update Best Model:",target_best_model, '*LOSS:', mvalid_smape, " ###")
            SAVE_PATH = args.save + "exp" + str(args.expid) + "_" + str(runid) +".pth"
            torch.save({
              'epoch': i,
              'task_level': engine.task_level,
              'model_state_dict': engine.model.state_dict(),
              'optimizer_state_dict': engine.optimizer.state_dict(),
              'loss': mvalid_mae,
              'train_loss': train_loss_epoch,
              'valid_loss': valid_loss_epoch
            }, SAVE_PATH)
            minl = mvalid_mae

    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))


    bestid = np.argmin(his_loss)
    

    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid],4)))
    
    target_model = args.save + "exp" + str(args.expid) + "_" + str(runid) +".pth"
   
    print("### loading model is:",target_model ,'###')
    
    
    SAVE_PATH = args.save + "exp" + str(args.expid) + "_" + str(runid) +".pth"
    
    checkpoint = torch.load(SAVE_PATH)
    engine.model.load_state_dict(checkpoint['model_state_dict'])
    engine.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    engine.task_level = checkpoint['task_level']
    start_epoch = checkpoint['epoch']
    train_loss_epoch = checkpoint['train_loss']
    valid_loss_epoch = checkpoint['valid_loss']
    

    ### 測試讀取出的model ### 
    valid_mae = []  
    valid_mape = [] 
    valid_smape = []    
    valid_rmse = [] 
    tmp_y = []
    for iter, (x, y,x_f1,x_f2,x_f3,x_f4)  in enumerate(dataloader['val_loader'].get_iterator()):  
        
        testx = torch.Tensor(x).to(device)  
        testx = testx.transpose(1, 3)   
        testy = torch.Tensor(y).to(device)  
        testy = testy.transpose(1, 3)   
        
        testx_f1 = torch.Tensor(x_f1).to(device)
        testx_f1= testx_f1.transpose(1, 3)

        testx_f2 = torch.Tensor(x_f2).to(device)
        testx_f2= testx_f2.transpose(1, 3)

        testx_f3 = torch.Tensor(x_f3).to(device)
        testx_f3= testx_f3.transpose(1, 3)
        
        testx_f4 = torch.Tensor(x_f4).to(device)
        testx_f4= testx_f4.transpose(1, 3)

        metrics = engine.eval(testx, testx_f1,testx_f2, testx_f3,testx_f4,testy[:,0,:,:]) 
        valid_mae.append(metrics[0])    
        valid_mape.append(metrics[1])   
        valid_rmse.append(metrics[2])   
        valid_smape.append(metrics[3])

    mvalid_mae = np.mean(valid_mae) 
    mvalid_mape = np.mean(valid_mape)   
    mvalid_rmse = np.mean(valid_rmse)   
    mvalid_smape = np.mean(valid_smape) 
    print("### 2-The valid loss on loding model is", str(round(mvalid_smape,4)))
    minl= valid_smape   
    print("### minl:",minl, "checkpoint['loss']:",checkpoint['loss'])   
    ### 測試讀取出的model ### 

    #valid data
    outputs = []
    realy = torch.Tensor(dataloader['y_val']).to(device)
    
    realy = realy.transpose(1,3)[:,0,:,:]
    print('#realy', realy.shape)
    
    for iter, (x, y,x_f1,x_f2,x_f3,x_f4)  in enumerate(dataloader['val_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1,3)
        
        testx_f1 = torch.Tensor(x_f1).to(device)
        testx_f1= testx_f1.transpose(1, 3)

        testx_f2 = torch.Tensor(x_f2).to(device)
        testx_f2= testx_f2.transpose(1, 3)

        testx_f3 = torch.Tensor(x_f3).to(device)
        testx_f3= testx_f3.transpose(1, 3)

        testx_f4 = torch.Tensor(x_f4).to(device)
        testx_f4= testx_f4.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx,testx_f1,testx_f2,testx_f3,testx_f4)
            preds = preds.transpose(1,3)  # 64,1,6,12

        outputs.append(preds.squeeze()) # 64,1,6,12 ->squeeze()->64,6,12

    yhat = torch.cat(outputs,dim=0)
    yhat = yhat[:realy.size(0),...]  # 5240,6,12
    print('# cat valid preds', yhat.shape)

    pred = dataloader['scaler'].inverse_transform(yhat)
    
    vmae, vmape, vrmse,vsmape = metric(pred,realy)
    print("valid - vmae, vmape, vrmse,vsmape", vmae, vmape, vrmse,vsmape)
    #----------------------------------#
    #test data
    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y,x_f1,x_f2,x_f3,x_f4)  in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        
        testx_f1 = torch.Tensor(x_f1).to(device)
        testx_f1= testx_f1.transpose(1, 3)

        testx_f2 = torch.Tensor(x_f2).to(device)
        testx_f2= testx_f2.transpose(1, 3)

        testx_f3 = torch.Tensor(x_f3).to(device)
        testx_f3= testx_f3.transpose(1, 3)

        testx_f4 = torch.Tensor(x_f4).to(device)
        testx_f4= testx_f4.transpose(1, 3)

        
        with torch.no_grad():
            preds = engine.model(testx,testx_f1,testx_f2,testx_f3,testx_f4)
            preds = preds.transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]  #10478, 6, 12
    print('# cat test preds', yhat.shape)
    
    mae = []
    mape = []
    rmse = []
    smape = []
    
    for i in range(args.seq_out_len):
        pred = dataloader['scaler'].inverse_transform(yhat[:, :, i])
        
        real = realy[:, :, i]

        metrics = metric(pred, real)
        
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}, Test SMAPE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2], metrics[3]))
        mae.append(metrics[0])
        mape.append(metrics[1])
        rmse.append(metrics[2])
        smape.append(metrics[3])
        
    #sys.exit()
    log = '{:.2f}   {:.2f}  {:.4f}  {:.4f}  '
    print("#### Final Results:")
    print(  str(args.expid) + "_" + str(runid)+'    ', 
          log.format(mae[0], rmse[0], smape[0], mape[0]),
          log.format(mae[2], rmse[2], smape[2], mape[2]),
          log.format(mae[5], rmse[5], smape[5], mape[5]),
          log.format(mae[11], rmse[11], smape[11], mape[11]),
         )
    ### Drawing Loss Diagram ###
    fig = plt.figure(figsize=(10, 6), dpi=600)
    plt.plot(checkpoint['train_loss'], label="train loss")
    plt.plot(checkpoint['valid_loss'], label="valid loss")
    plt.legend(loc="upper right")
    plt.title('#Loss of Training', fontsize=20)
    plt.ylabel("MAE", fontsize=14)
    plt.xlabel("Epochs", fontsize=14)
    plt.show()
    return vmae, vmape, vrmse,vsmape, mae, mape, rmse, smape

if __name__ == "__main__":

    vmae = []
    vmape = []
    vrmse = []
    vsmape = []
    mae = []
    mape = []
    rmse = []
    smape = []
    for i in range(args.runs):
        vm1, vm2, vm3, vm4, m1, m2, m3, m4 = main(i)
        vmae.append(vm1)
        vmape.append(vm2)
        vrmse.append(vm3)
        vsmape.append(vm4)
        mae.append(m1)
        mape.append(m2)
        rmse.append(m3)
        smape.append(m4)

    mae = np.array(mae)
    mape = np.array(mape)
    rmse = np.array(rmse)
    smape = np.array(smape)

    amae = np.mean(mae,0)
    amape = np.mean(mape,0)
    armse = np.mean(rmse,0)
    asmape = np.mean(smape,0)

    smae = np.std(mae,0)
    s_mape = np.std(mape,0)
    srmse = np.std(rmse,0)
    ssmape = np.std(smape,0)

    print('\n\nResults for runs\n\n')
    #valid data
    print('valid\tMAE\tRMSE\tMAPE')
    log = 'mean:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.mean(vmae),np.mean(vrmse),np.mean(vmape),np.mean(vsmape)))
    log = 'std:\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.std(vmae),np.std(vrmse),np.std(vmape),np.std(vsmape)))
    print('\n\n')
    #test data
    print('test|horizon\tMAE-mean\tRMSE-mean\tMAPE-mean\tMAE-std\tRMSE-std\tMAPE-std')
    
    for i in [2,5,11]:
        log = '{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'
        print(log.format(i+1, amae[i], armse[i], amape[i], asmape[i], smae[i], srmse[i], s_mape[i], ssmape[i]))