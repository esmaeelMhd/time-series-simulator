# Retrainer, start_retrain() method
if self.retrain_args.retrain_method == 'v1':
    for i, (data_item, target_item) in enumerate(zip(self.train_data, self.train_targets)):
        if experiment == 3 or experiment == 4:
            batch_idx = self.data_idxs[i]
        else:
            batch_idx = i
            
        batch_data = torch.Tensor(np.array(data_item)).unsqueeze(0).to(self.device)
        batch_targets = torch.Tensor(np.array(target_item)).unsqueeze(0).to(self.device)
        
        train_dataset = TensorDataset(batch_data, batch_targets)
        train_loader = DataLoader(train_dataset, batch_size=1, 
                                  shuffle=False, drop_last=True)
        
        with torch.cuda.amp.autocast():
            training_loss, validation_loss, self.loaded_model = self.opt.self_supervised_train(model=self.loaded_model,
                                                                   train_loader=train_loader,
                                                                   val_loader=self.val_loader,
                                                                   epochs=self.retrain_args.retrain_epochs,
                                                                   use_actual_actions=self.retrain_args.use_actual_actions,
                                                                   retrain_mode=self.retrain_args.retrain_mode,
                                                                   scheduled_sampling=self.retrain_args.scheduled_sampling,
                                                                   scheduled_sampling_prob=self.retrain_args.ss_prob)
        # dict_t = self.opt.retrain_dict
        # self.train_dictionary.append(dict_t)
        self.train_losses.append(training_loss)
        self.val_losses.append(validation_loss)
        train_loss = np.mean(self.train_losses)
        
        if self.best_loss is None or self.best_loss > validation_loss:
            self.best_loss = validation_loss
            print('Saving the model ...')
            torch.save(self.loaded_model.state_dict(), checkpoint_path + '/' + self.retrain_name + '.pth')
            
        print(f'Ex {experiment}: [Batch {i+1}/{len(self.train_data)}] | Idx: {batch_idx} | EL: {batch_targets.shape[1]}'+\
              f' | Loss: {validation_loss:.7f} | Best loss: {self.best_loss:.7f} | Train loss : {train_loss:.7f}')

# LSTM_model_optimizer
def self_supervised_train(self, model, train_loader, val_loader, epochs=10,
                          use_actual_actions=False, retrain_mode='single_pred',
                          scheduled_sampling=False, scheduled_sampling_prob=0.5):
    # Start training
    self.model = model
    epoch_losses = []
    '''
    self.retrain_dict = {'x':[],
                        'targets':[],
                        'val_x':[],
                        'y_val':[],
                        'y_pred':[],
                        'val_losses':[]}
    '''
    
    for epoch in range(1, epochs + 1):
        for x, targets in train_loader:
            x = x.to(self.device)
            targets = targets.to(self.device)
            # print('first x shape: ', x.shape)
            # print('targets shape: ', targets.shape)
            prev_pred = None
            losses = []
            sum_losses = 0
            
            if retrain_mode == 'single_pred':
                for i in range(targets.shape[1]):
                    self.model.train()
                    y_true = targets[:,i,:].to(self.device)
                    if prev_pred is not None:
                        prev_pred = torch.cat((prev_pred, y_true[:, self.args.out_features:]), dim=1)
                        # print('prev_pred shape: ', prev_pred.shape)
                        x = torch.cat([x, prev_pred.unsqueeze(0)], dim=1).to(self.device)
                        x = x[:, 1:, :]
                    # print('second x shape: ', x.shape)
                    # print('y pred shape: ', y_pred.shape)
                    # print('y true shape: ', y_true.shape)
                    loss = self.train_step(x, y_true[:,:self.args.out_features])                  
                    losses.append(loss)    
                    sum_losses += loss
                    
                    with torch.no_grad(), torch.cuda.amp.autocast():
                        self.model.eval()
                        prev_pred = self.model(x)[:, 0, :].view(1, self.args.out_features)
                        # print('prev_pred: ', prev_pred.shape)
                        if use_actual_actions:
                            prev_pred[:, 0] = y_true[:, 0]
                                    
                # Losses of the whole episode
                epoch_loss = np.mean(losses)
                epoch_losses.append(epoch_loss)
            
            elif retrain_mode == 'batch_pred':
                if (scheduled_sampling == True and torch.rand(1) >= scheduled_sampling_prob) or scheduled_sampling == False:
                    batch_size = targets.shape[1]
                    x_list = []
                    for i in range(batch_size):
                        x_list.append(x)
                        y_true = targets[:,i,:].to(self.device)
                        # print('y_true shape: ', y_true.shape)
                        with torch.no_grad(), torch.cuda.amp.autocast():
                            self.model.eval()
                            pred = self.model(x)[:, 0, :].view(1, self.args.out_features)

                            if use_actual_actions:
                                pred[:, 0] = y_true[:, 0]
                            
                            pred = torch.cat((pred, y_true[:, self.args.out_features:]), dim=1)
                            # print('prev_pred shape: ', prev_pred.shape)
                            x = torch.cat([x, pred.unsqueeze(0)], dim=1).to(self.device)
                            x = x[:, 1:, :]
                            
                    x = torch.stack(x_list).view([batch_size, -1, self.args.in_features]).to(self.device)
                    # squeeze(0) to convert the shape to (batch_size, out_features)
                    targets = targets[:, :, :self.args.out_features].squeeze(0)
                                        
                else:
                    x_list = []
                    batch_size = targets.shape[1]
                    x = torch.cat([x, targets], dim=1).to(self.device).squeeze(0)
                    for i in range(batch_size):
                        x_list.append(x[i:i+self.args.seq_len,:])
                    x = torch.stack(x_list).view([batch_size, -1, self.args.in_features]).to(self.device)
                    targets = targets[:, :, :self.args.out_features].squeeze(0)
                    
                # print(x.shape)
                # print(targets.shape)
                # self.retrain_dict['x'].append(np.array(x.detach().cpu()))
                # self.retrain_dict['targets'].append(np.array(targets.detach().cpu()))
                loss = self.train_step(x, targets)
                epoch_losses.append(loss)
            
            val_loss = self.val_retrain(self.model, val_loader, use_actual_actions)
    
    # Loss of the batch for all epochs
    training_loss = np.mean(epoch_losses)
    self.train_losses.append(training_loss)
    
    return training_loss, val_loss, self.model