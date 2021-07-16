import json
import numpy as np
import os
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
import torch

class TrajectoryDataset(Dataset):

    def __init__(self, mainpath, part):
        super(TrajectoryDataset, self).__init__()
        self.part = part
        self.mainpath = os.path.join(mainpath, part)
        self.names = sorted(os.listdir(self.mainpath))
        self.obs = 20
        self.preds = 80
        self.total = self.obs+self.preds
        if self.part=='test':
            self.total = self.obs
        self.type_to_int = {'vehicle':0, 'cyclist':1, 'human':2, 'unknown':3}

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        with open(os.path.join(self.mainpath, self.names[index]), 'r') as inp:
            data = json.load(inp)

        out = {'coords':np.empty( (0,self.obs,2), dtype=np.float32),
               'heading':np.empty( (0,self.obs,1), dtype=np.float32),
               'future_coords':np.empty( (0,self.preds,2), dtype=np.float32),
               'future_heading':np.empty( (0,self.preds,1), dtype=np.float32),
               'type':[], 'scene_id':[], 'track_id':[]}

        for track_id, tmp in data['AGENTS'].items():
            current_type = self.type_to_int[tmp['type']]

            if len(tmp['visible'])==self.total and current_type!=3:
                coords = np.array(tmp['coords'], dtype=np.float32)
                out['coords'] = np.vstack(( out['coords'], coords[:self.obs].reshape(1,self.obs,2) ))

                heading = np.array(tmp['heading'], dtype=np.float32)
                out['heading'] = np.vstack(( out['heading'], heading[:self.obs].reshape(1,self.obs,1) ))

                if not self.part=='test':
                    out['future_coords'] = np.vstack(( out['future_coords'],
                                            coords[self.obs:].reshape(1,self.preds,2) ))
                    out['future_heading'] = np.vstack(( out['future_heading'],
                                            heading[self.obs:].reshape(1,self.preds,1) ))

                out['type'].append( current_type )
                out['scene_id'].append( int(self.names[index][:-5]) )
                out['track_id'].append( int(track_id) )

        out['type'] = np.array(out['type']).reshape(-1,1)
        out['scene_id'] = np.array(out['scene_id']).reshape(-1,1)
        out['track_id'] = np.array(out['track_id']).reshape(-1,1)

        if self.part=='test':
            out.pop('future_coords')
            out.pop('future_heading')

        return out

    def show_scene(self, index):
        with open(os.path.join(self.mainpath, self.names[index]), 'r') as inp:
            data = json.load(inp)

        plt.figure(figsize=(10,10))
        ax = plt.gca()
        human_color = next(ax._get_lines.prop_cycler)['color']
        unknown_color = next(ax._get_lines.prop_cycler)['color']

        for track_id, tmp in data['AGENTS'].items():
            tmp['visible'] = np.array(tmp['visible'])
            tmp['coords'] = np.array(tmp['coords'])

            current_color = next(ax._get_lines.prop_cycler)['color']
            for part in range(20):
                tmp2 = tmp['coords'][(tmp['visible']>=part*5) & (tmp['visible']<=(part+1)*5)]
                if tmp['type'] in ['vehicle','cyclist']:
                    plt.plot(tmp2[:,0], tmp2[:,1], current_color, alpha=(0.025*part)+0.4, linewidth=part/4.+3)
                elif tmp['type']=='human':
                    plt.plot(tmp2[:,0], tmp2[:,1], human_color, alpha=(0.025*part)+0.2, linewidth=part/8.+1, linestyle='--')
                elif tmp['type']=='unknown':
                    plt.plot(tmp2[:,0], tmp2[:,1], unknown_color, alpha=(0.025*part)+0.2, linewidth=part/8.+1, linestyle='--')
        plt.axis('square')
        plt.show()

def collate_fn(batch):
    out = {}
    for k in batch[0].keys():
        out[k] = np.vstack( [x[k] for x in batch] )
    return out

def torch_collate_fn(batch):
    out = {}
    for k in batch[0].keys():
        out[k] = torch.Tensor(np.vstack( [x[k] for x in batch] ))
    return out
