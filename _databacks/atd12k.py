
from _util.util_v0 import * ; import _util.util_v0 as uutil
from _util.twodee_v0 import * ; import _util.twodee_v0 as u2d
from _util.pytorch_v0 import * ; import _util.pytorch_v0 as utorch

class DatabackendATD12k:
    def __init__(self):
        self.dn = '/home/jiaming/atd12k_points'
        self.fn = '/home/jiaming/sgm'
        self.test_source = '540p'
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
        flow0 = torch.from_numpy(load('{}/{}/guide_flo13.npy'.format(self.fn, self.get_ffn(bn)))).flip(dims=(0, 1))
        flow1 = torch.from_numpy(load('{}/{}/guide_flo31.npy'.format(self.fn, self.get_ffn(bn)))).flip(dims=(0, 1))
        return {
            'bn': bn,
            'images': [
                I(self.get_fn(bn, i))
                for i in range(3)
            ],
            'flows': (flow0, flow1)
        }

    def get_ffn(self, bn):
        tt, tid = bn.split('/')
        if tt=='test':
            fff = 'test_2k_540p'
        else:
            fff = 'train_10k'
        return '{}/{}'.format(fff, tid)


    def get_fn(self, bn, fidx):
        tt,tid = bn.split('/')
        if tt=='test':
            dn = '{}/test_2k_{}'.format(self.dn, self.test_source)
            # ext = 'png' if self.test_source=='540p' else 'jpg'
            ext = 'jpg'
        else:
            dn = '{}/train_10k'.format(self.dn)
            ext = 'jpg'
        return '{}/{}/frame{}.{}'.format(dn,tid,fidx+1,ext)
    def get_bns(self):
        return sorted([
            'test/{}'.format(dn)
            for dn in os.listdir('{}/test_2k_{}'.format(self.dn, self.test_source))
            if os.path.isdir('{}/test_2k_{}/{}'.format(self.dn, self.test_source, dn))
        ])

