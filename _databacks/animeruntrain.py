
from _util.util_v0 import * ; import _util.util_v0 as uutil
from _util.twodee_v0 import * ; import _util.twodee_v0 as u2d
from _util.pytorch_v0 import * ; import _util.pytorch_v0 as utorch

class DatabackendAnimeRunTrain:
    def __init__(self):
        self.dn = '/home/kuhu6123/jshe2377/AnimeRun/AnimeRun/'
        self.fn = '/home/kuhu6123/jshe2377/AnimeRun/AnimeRun'
        # self.test_source = '540p'
        self.bns = np.array(self.get_bns(), dtype=np.string_)
        return
    def __len__(self):
        return len(self.bns)
    def __getitem__(self, idx):
        if isinstance (idx, int):
            bn = str(self.bns[idx], encoding='utf-8')
        elif isinstance(idx, str):
            bn = idx
        else:
            assert 0, '{} not understood'.format(idx)
        tt, tid = bn.split('/')
        flow0 = torch.from_numpy(load('{}/{}/guide_flo13.npy'.format(self.fn, self.get_ffn(bn))))
        flow1 = torch.from_numpy(load('{}/{}/guide_flo31.npy'.format(self.fn, self.get_ffn(bn))))
        # print(flow1.size())
        return {
            'bn': bn,
            'images': [
                I(self.get_fn(bn, i))
                for i in range(3)
            ],
            'flows': torch.stack([flow0,flow1], dim=1),
            'fn': tid
        }

    def get_ffn(self, bn):
        tt, tid = bn.split('/')
        if tt=='test':
            fff = 'test'
        else:
            fff = 'train'
        return '{}/contour_region/{}'.format(fff, tid)


    def get_fn(self, bn, fidx):
        tt,tid = bn.split('/')
        if tt=='test':
            dn = '{}/test/contourcopy'.format(self.dn)
            ext = 'jpg'
        else:
            dn = '{}/train/contourcopy'.format(self.dn)
            ext = 'jpg'
        return '{}/{}/frame{}.{}'.format(dn,tid,fidx+1,ext)
    def get_bns(self):
        return sorted([
            'test/{}'.format(dn)
            for dn in os.listdir('{}/test/contourcopy'.format(self.dn))
            if os.path.isdir('{}/test/contourcopy/{}'.format(self.dn, dn))
        ])

