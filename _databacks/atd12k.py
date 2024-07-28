
from _util.util_v0 import * ; import _util.util_v0 as uutil
from _util.twodee_v0 import * ; import _util.twodee_v0 as u2d
from _util.pytorch_v0 import * ; import _util.pytorch_v0 as utorch
from torchvision import transforms


class DatabackendATD12k:
    def __init__(self, is_training):
        self.data_root = "/share/hhd3/kuhu6123/atd12k_points/atd12k_points"
        self.region_root = "/share/hhd3/kuhu6123/atd12k_points/atd12k_points"
        self.training = is_training
        if is_training:
            self.region_root = os.path.join(self.data_root, 'train_10k_region')
            self.data_root = os.path.join(self.data_root, 'train_10k')
        else:
            self.region_root = os.path.join(self.data_root, 'test_2k_region')
            self.data_root = os.path.join(self.data_root, 'test_2k_540p')


        dirs = os.listdir(self.data_root)
        data_list = []
        for d in dirs:
            if d == '.DS_Store':
                continue
            img0 = os.path.join(self.data_root, d, 'frame1.jpg')
            img1 = os.path.join(self.data_root, d, 'frame3.jpg')

            gt = os.path.join(self.data_root, d, 'frame2.jpg')

            region13 = os.path.join(self.region_root, d, 'guide_flo13.npy')
            region31 = os.path.join(self.region_root, d, 'guide_flo31.npy')

            # data_list.append([img0, img1, points14, points12, points34, gt, d])
            data_list.append([img0, img1, gt, d, region13, region31])

        self.data_list = data_list

        if self.training:
            self.transforms = transforms.Compose([
                # transforms.RandomCrop(228),
                transforms.RandomHorizontalFlip(),
                # transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                transforms.ToTensor()
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor()
            ])

    def __len__(self):
        if self.training:
            return len(self.data_list)
        else:
            return len(self.data_list)

    def __getitem__(self, index):
        imgpaths = [self.data_list[index][0], self.data_list[index][1], self.data_list[index][2]]
        images = [Image.open(pth) for pth in imgpaths]

        size = (384, 192)
        flow13 = np.load(self.data_list[index][4]).astype(np.float32)
        flow31 = np.load(self.data_list[index][5]).astype(np.float32)
        flow = [flow13, flow31]
        if self.training:
            seed = random.randint(0, 2 ** 32)
            images_ = []
            for img_ in images:
                random.seed(seed)
                images_.append(self.transforms(img_))
            images = images_

            gt = images[2]

            images = images[:2]
            imgpath = self.data_list[index][3]


            # return images, gt, flow
        else:
            T = self.transforms
            images = [T(img_.resize(size)) for img_ in images]

            gt = images[2]
            images = images[:2]
            imgpath = self.data_list[index][3]
            # return images, gt, imgpath, flow

        return {
            'bn': imgpath,
            'images': images,
            'flows': torch.stack([flow13,flow31], dim=1),
            'fn': imgpath
        }

    def get_ffn(self, bn):
        tt, tid = bn.split('/')
        if tt=='test':
            fff = 'test_2k_region'
        else:
            fff = 'train_10k_region'
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

