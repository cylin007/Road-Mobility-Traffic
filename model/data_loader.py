

"""### Loading Data: Univariate+MA+GA"""

"""### Loading Data"""

batch_size = args.batch_size
valid_batch_size = args.batch_size
test_batch_size = args.batch_size
data = {}

augmented_features = ['','_ma3','_ma6','_ga12','_ga24']
key_name = ['','_f1','_f2','_f3','_f4']

for i in range(len(augmented_features)):
    print("range_type", augmented_features[i])
    for category in ['train', 'val', 'test']:
        key = category + key_name[i]
        key2 = category + augmented_features[i]
        # Loading npz 
        cat_data = np.load(os.path.join(args.data, key2 + '.npz'))
        print("loading... key:", key ,'->', args.data, key2 + '.npz')

        if category == "train":
          data['x_' + key] = cat_data['x'][:]     # (?, 12, 207, 2)
          data['y_' + key] = cat_data['y'][:]   # (?, 12, 207, 2)
        else:
          data['x_' + key] = cat_data['x']     # (?, 12, 207, 2)
          data['y_' + key] = cat_data['y']     # (?, 12, 207, 2)
    
    print(data.keys())
    if augmented_features[i] == '':
        # 使用train的mean/std來正規化valid/test #
        scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
        data['scaler'] = scaler
        
    # 將欲訓練特徵改成正規化
    for category in ['train', 'val', 'test']:
        key = category + key_name[i]
        data['x_' + key][..., 0] = data['scaler'].transform(data['x_' + key][..., 0])
        print("data['x_' + key]:", 'x_' + key)

data['train_loader'] = DataLoaderM(
    data['x_train'], data['y_train'], 
    data['x_train_f1'], 
    data['x_train_f2'], 
    data['x_train_f3'],
    data['x_train_f4'],
    batch_size)

data['val_loader'] = DataLoaderM(
    data['x_val'], data['y_val'], 
    data['x_val_f1'],  
    data['x_val_f2'],  
    data['x_val_f3'],  
    data['x_val_f4'],  
    valid_batch_size)

data['test_loader'] = DataLoaderM(
    data['x_test'], data['y_test'], 
    data['x_test_f1'],  
    data['x_test_f2'], 
    data['x_test_f3'],
    data['x_test_f4'],  
    test_batch_size)

print(data.keys())
'''
adj_mx: 根據distances_la_2012.csv, 找出每個sensor與其他sensor距離並建立距離矩陣, 再進行正規化
'''
sensor_ids, sensor_id_to_ind, adj_mx = load_adj(args.adj_data,args.adjtype)   # adjtype: default='doubletransition'

adj_mx = [torch.tensor(i).to(device) for i in adj_mx]

dataloader = data.copy()