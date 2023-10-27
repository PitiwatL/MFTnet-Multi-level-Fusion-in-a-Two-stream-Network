import numpy as np
import torch

def callbacks(eph, model, loss_epoch_val, val_loss_his, count, SaveWeight_path):
    if np.mean(loss_epoch_val) <= min(val_loss_his):
        if eph > 0 :
            print(f'Loss Validation improves from {val_loss_his[eph-1]:.7f} to {val_loss_his[eph]:.7f}')
            torch.save(model.state_dict(), SaveWeight_path)
            print('Save best weight!')
            
            return 'Continue'
            
    if np.mean(loss_epoch_val) > min(val_loss_his) :
        print(f'Loss Validation does not improve from {min(val_loss_his):.7f}')
        print(f'patience ..... {count} .....')
        
        if count == 20:
            print(f'stop training!')
            
            return 'Break'
        
        return 'not_improve'
        
    if (np.mean(loss_epoch_val) < min(val_loss_his)) and (np.mean(loss_epoch_val) - min(val_loss_his)) < 0.0001 :
        print(f'Loss Validation does not improve from {min(val_loss_his):.7f}')
        print(f'patience ..... {count} .....')
        if count == 20:
            print(f'stop training!')
            
            return 'Break'
        
        return 'not_improve'