import torch
from config import DEVICE as device
from tqdm.auto import tqdm
from utils import LrScheduler, EarlyStopping, SaveBestModel
import numpy as np

def fit(model, train_loader, optimizer, criterion):
    loss_list = []
        
    prog_bar = tqdm(enumerate(train_loader), total=len(train_loader), leave=True, desc="Training Set: ")
    for i, (source, target, attention_mask) in prog_bar:
        # load data and labels to device
        source = source.to(device)
        target = target.to(device)
        attention_mask = attention_mask[:, :-1].to(device)
        target_input = target[:, :-1].clone()
        out = model(source, target_input, attention_mask)
        out = out.reshape(-1, out.shape[2])
        target_out = target[:, 1:].reshape(-1)
        optimizer.zero_grad()
        loss = criterion(out, target_out)
        
        if i%10000==0 and i !=0:
            torch.save({'model_state_dict' : model.state_dict()}, f'bestmodel_{i}.pth')

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        loss_list.append(loss.item())
        prog_bar.set_description(f"Batch : {i+1}/{len(train_loader)}")
        prog_bar.set_postfix(loss=loss.item())

    train_loss = np.mean(loss_list) 
    
    return train_loss






def validate(model, val_loader, criterion):
    
    with torch.no_grad():
        val_losses = []
        prog_bar = tqdm(enumerate(val_loader), total=len(val_loader), leave=True, desc="Validation Set: ")
        for i, (source, target, attention_mask) in prog_bar:
            source = source.to(device)
            target = target.to(device)
            attention_mask = attention_mask[:, :-1].to(device)
            target_input = target[:, :-1].clone()
            out = model(source, target_input, attention_mask)
            out = out.reshape(-1, out.shape[2])
            target_out = target[:, 1:].reshape(-1)
            loss = criterion(out, target_out)
            val_losses.append(loss.item())
            prog_bar.set_description(f"Batch : {i+1}/{len(val_loader)}")
            prog_bar.set_postfix(loss=loss.item())

        val_loss = np.mean(val_losses)
    return val_loss



def model_train(model, train_loader, val_loader, num_epochs, learning_rate, criterion, optimizer, early_stop=False):
    train_loss_list = []
    save_best_model = SaveBestModel("bestmodel")
    lr_scheduler = LrScheduler(optimizer, patience=1)
    if early_stop:
        early_stopping = EarlyStopping(patience=10)

    for epoch in range(num_epochs):

        train_loss = fit(model, train_loader, optimizer, criterion)
        save_best_model(train_loss, epoch, model, optimizer, criterion)
        val_loss = validate(model, val_loader, criterion)
        train_loss_list.append(train_loss)

        print(f"""Training Set :\nEpoch :{epoch+1}/{num_epochs}, \tloss : {train_loss:.3f},\tLearning Rate : {learning_rate}""")
        print(f"""Validation Set :\nEpoch :{epoch+1}/{num_epochs}, \tloss : {val_loss:.3f}""")
        lr_scheduler(train_loss)
        learning_rate = optimizer.param_groups[0]["lr"]
        print("--"*40)
        if early_stopping:
            early_stopping(train_loss)
            if early_stopping.early_stop:
                break

    return model, train_loss_list, val_loss