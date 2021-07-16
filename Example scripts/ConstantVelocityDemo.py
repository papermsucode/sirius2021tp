import numpy as np
from tqdm import tqdm
import json

from torch.utils.data import DataLoader
import torch.nn as nn

from iterator import TrajectoryDataset, collate_fn

class Predictor(nn.Module):

    def __init__(self, encoder, interactor, forecaster):
        super(Predictor, self).__init__()

        self.encoder = encoder
        self.interactor = interactor
        self.forecaster = forecaster

    def forward(self, batch):
        encoder_embs = self.encoder(batch)
        interactor_embs = self.interactor(batch, encoder_embs)
        preds = self.forecaster(batch, interactor_embs)
        return preds

class NoneEncoder():
    def __call__(self, batch):
        return None

class NoneInteractor():
    def __call__(self, batch, encoder_embs):
        return None

class ConstantForecaster():
    def __init__(self, use_acceleration=False, aver_vel_num=5, aver_acc_num=10):
        self.use_acceleration = use_acceleration
        self.aver_vel_num = aver_vel_num
        self.aver_acc_num = aver_acc_num
        assert self.aver_vel_num>0 and self.aver_vel_num<=20
        assert self.aver_acc_num>0 and self.aver_acc_num<20

    def __call__(self, batch, interactor_embs):
        velocity = batch['coords'][:,1:] - batch['coords'][:,:-1]
        acceleration = velocity[:,1:] - velocity[:,:-1]

        mean_velocity = np.mean(velocity[:,-self.aver_vel_num:], 1, keepdims=True)
        mean_acceleration = np.mean(acceleration[:,-self.aver_acc_num:], 1, keepdims=True)
        steps = np.arange(1,81).reshape(1,-1,1)

        preds = batch['coords'][:,-1:] + mean_velocity*steps
        if self.use_acceleration:
            preds += mean_acceleration*steps**2/2
        return preds

if __name__=='__main__':
    ConstantModel = Predictor( NoneEncoder(), NoneInteractor(), ConstantForecaster() )

    data = TrajectoryDataset('Sirius_json/','val')
    data_loader = DataLoader(
        data,
        batch_size=32,
        num_workers=2,
        collate_fn=collate_fn,
        shuffle=False,
    )

    gt_short = []
    gt_medium = []
    gt_long = []

    preds_short = []
    preds_medium = []
    preds_long = []

    print('Evaluation on the validation set')

    for batch in tqdm(data_loader):
        out = ConstantModel(batch)

        gt_short.append( batch['future_coords'][:,:20] )
        gt_medium.append( batch['future_coords'][:,:40:2] )
        gt_long.append( batch['future_coords'][:,:80:4] )

        preds_short.append( out[:,:20] )
        preds_medium.append( out[:,:40:2] )
        preds_long.append( out[:,:80:4] )

    errors = np.sqrt(np.sum((np.concatenate(gt_short)-np.concatenate(preds_short))**2, 2))
    print(f'Short:\t\tADE={np.mean(errors)}\t{np.mean(errors[:,-1])}')

    errors = np.sqrt(np.sum((np.concatenate(gt_medium)-np.concatenate(preds_medium))**2, 2))
    print(f'Medium:\tADE={np.mean(errors)}\t{np.mean(errors[:,-1])}')

    errors = np.sqrt(np.sum((np.concatenate(gt_long)-np.concatenate(preds_long))**2, 2))
    print(f'Long:\t\tADE={np.mean(errors)}\t{np.mean(errors[:,-1])}')

    with open('Sirius_json/test_agents.json') as f:
        test = json.load(f)

    results = {k:{} for k in test.keys()}

    data = TrajectoryDataset('Sirius_json/','test')
    data_loader = DataLoader(
        data,
        batch_size=32,
        num_workers=2,
        collate_fn=collate_fn,
        shuffle=False,
    )

    print('Predicting and preparation of the submission file')

    for batch in tqdm(data_loader):
        out = ConstantModel(batch)

        preds_short = out[:,:20]
        preds_medium = out[:,:40:2]
        preds_long = out[:,:80:4]

        for i, (scene,agent) in enumerate(zip(batch['scene_id'][:,0],batch['track_id'][:,0])):
            if agent in test[str(scene)]:
                results[str(scene)][int(agent)] = {}
                results[str(scene)][int(agent)]['short'] = preds_short[i].tolist()
                results[str(scene)][int(agent)]['medium'] = preds_medium[i].tolist()
                results[str(scene)][int(agent)]['long'] = preds_long[i].tolist()

    with open('ConstantVelocitySubmit.json','w') as f:
        json.dump(results, f)
