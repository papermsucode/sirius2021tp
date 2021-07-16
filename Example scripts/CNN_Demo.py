import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from time import time
import json
from tqdm import tqdm

from iterator import TrajectoryDataset, torch_collate_fn

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

class CNN_encoder(nn.Module):

    def __init__(self, input_size=19, input_dim=2, widths=[64,64,128,128,256], strides=[1,1,2,2,1], dropout=0.2):
        super(CNN_encoder, self).__init__()
        assert len(widths)==len(strides)

        layers = []
        widths = [input_dim]+widths
        for i in range(len(widths)-1):
            layers.extend([nn.Conv1d(widths[i],widths[i+1],3,stride=strides[i]), nn.ReLU(True), nn.BatchNorm1d(widths[i+1])])

        self.model = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

        remain_size = input_size
        for s in strides:
            if s==1:
                remain_size -= 2
            elif s==2:
                remain_size = (remain_size-1)//2

        self.fc_size = remain_size*widths[-1]
        print('Output size is',self.fc_size)

    def forward(self, batch):
        x = (batch['coords'][:,1:] - batch['coords'][:,:-1]).transpose(1,2)
        x = self.model(x)
        x = x.view(-1, self.fc_size).contiguous()
        return self.dropout(x)

class pass_interactor(nn.Module):
    def forward(self, batch, encoder_embs):
        return encoder_embs

class MLP_Forecaster(nn.Module):

    def __init__(self, input_size=256, fc1_size=512, fc2_size=2*40,  dropout=0.2):
        super(MLP_Forecaster, self).__init__()
        self.fc1 = nn.Linear(input_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc2_size = fc2_size//2
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch, interactor_embs):
        x = self.fc1(interactor_embs)
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.view(-1,self.fc2_size,2)
        return batch['coords'][:,-1:]+x

def to_gpu(batch):
    for k in batch.keys():
        batch[k] = batch[k].cuda()
    return batch

if __name__=='__main__':
    NUM_OF_PREDS = 40

    train_data = TrajectoryDataset('Sirius_json','train')
    train_loader = DataLoader(
        train_data,
        batch_size=32,
        num_workers=2,
        collate_fn=torch_collate_fn,
        shuffle=True,
    )

    val_data = TrajectoryDataset('Sirius_json','val')
    val_loader = DataLoader(
        val_data,
        batch_size=64,
        num_workers=2,
        collate_fn=torch_collate_fn,
        shuffle=False,
    )

    CNN_Model = Predictor( CNN_encoder(), pass_interactor(), MLP_Forecaster(fc2_size=2*NUM_OF_PREDS) ).cuda()

    optimizer = optim.SGD(CNN_Model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[8,14], gamma=0.1)

    running_loss = 0.
    best_val = 100

    print('Start of training')

    for epoch in range(18):
        CNN_Model.train()

        t_data = 0
        t_cnn = 0
        t0 = time()
        for batch_idx, batch in enumerate(train_loader):
            t1 = time()
            t_data += t1-t0
            optimizer.zero_grad()
            batch = to_gpu(batch)
            preds = CNN_Model(batch)
            loss = torch.mean(F.pairwise_distance(preds.view(-1,2), batch['future_coords'][:,:NUM_OF_PREDS].contiguous().view(-1,2)))
            loss.backward()
            optimizer.step()
            t0 = time()
            t_cnn += t0-t1
            running_loss = 0.9*running_loss+0.1*loss
            if (batch_idx+1)%100==0 or batch_idx+1==len(train_loader):
                print('Epoch',epoch+1,'Step',batch_idx+1,\
                      'Loss',running_loss.item(),\
                      'Data time',t_data/(batch_idx+1),\
                      'CNN time',t_cnn/(batch_idx+1))

        CNN_Model.eval()
        total_ade = [0,0,0]
        total_fde = [0,0,0]
        count = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                batch = to_gpu(batch)
                preds = CNN_Model(batch)
                batch['coords'][:] = preds[:,NUM_OF_PREDS-20:]
                preds2 = CNN_Model(batch)
                preds = torch.cat(( preds,preds2 ),1)
                num_of_agents = preds.size()[0]

                total_ade[0] += num_of_agents*torch.mean(F.pairwise_distance(preds[:,:20].contiguous().view(-1,2), batch['future_coords'][:,:20].contiguous().view(-1,2))).item()
                total_ade[1] += num_of_agents*torch.mean(F.pairwise_distance(preds[:,:40:2].contiguous().view(-1,2), batch['future_coords'][:,:40:2].contiguous().view(-1,2))).item()
                total_ade[2] += num_of_agents*torch.mean(F.pairwise_distance(preds[:,:80:4].contiguous().view(-1,2), batch['future_coords'][:,:80:4].contiguous().view(-1,2))).item()

                total_fde[0] += torch.sum(F.pairwise_distance(preds[:,19].reshape(-1,2), batch['future_coords'][:,19].reshape(-1,2))).item()
                total_fde[1] += torch.sum(F.pairwise_distance(preds[:,39].reshape(-1,2), batch['future_coords'][:,39].reshape(-1,2))).item()
                total_fde[2] += torch.sum(F.pairwise_distance(preds[:,79].reshape(-1,2), batch['future_coords'][:,79].reshape(-1,2))).item()

                count += num_of_agents
        print('--------------------------------------')
        print('Epoch',epoch+1,'results on validation:')
        print('Short ADE',total_ade[0]/count,'Medium ADE',total_ade[1]/count,'Long ADE',total_ade[2]/count)
        print('Short FDE',total_fde[0]/count,'Medium FDE',total_fde[1]/count,'Long FDE',total_fde[2]/count)
        print('--------------------------------------')

        if best_val>total_ade[1]/count:
            best_val = total_ade[1]/count
            torch.save(CNN_Model.state_dict(),'best.ckpt')

        scheduler.step()

    with open('Sirius_json/test_agents.json') as f:
        test = json.load(f)

    results = {k:{} for k in test.keys()}

    data = TrajectoryDataset('Sirius_json/','test')
    data_loader = DataLoader(
        data,
        batch_size=64,
        num_workers=2,
        collate_fn=torch_collate_fn,
        shuffle=False,
    )

    print('Predicting and preparation of the submission file')

    CNN_Model.load_state_dict(torch.load('best.ckpt',map_location='cpu'))
    CNN_Model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader):
            batch = to_gpu(batch)
            preds = CNN_Model(batch)
            batch['coords'][:] = preds[:,NUM_OF_PREDS-20:]
            preds2 = CNN_Model(batch)
            preds = torch.cat(( preds,preds2 ),1)

            preds_short = preds[:,:20]
            preds_medium = preds[:,:40:2]
            preds_long = preds[:,:80:4]

            for i, (scene,agent) in enumerate(zip(batch['scene_id'][:,0],batch['track_id'][:,0])):
                scene = int(scene.item())
                agent = int(agent.item())
                if agent in test[str(scene)]:
                    results[str(scene)][agent] = {}
                    results[str(scene)][agent]['short'] = preds_short[i].tolist()
                    results[str(scene)][agent]['medium'] = preds_medium[i].tolist()
                    results[str(scene)][agent]['long'] = preds_long[i].tolist()

    with open('CNN_Submit.json','w') as f:
        json.dump(results, f)
