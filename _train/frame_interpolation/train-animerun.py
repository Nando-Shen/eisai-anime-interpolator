



import sys
import os
if sys.path[0]!='': sys.path.insert(0, '')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from _util.util_v0 import * ; import _util.util_v0 as util
from _util.twodee_v0 import * ; import _util.twodee_v0 as u2d
from _util.pytorch_v0 import * ; import _util.pytorch_v0 as utorch


ap = argparse.ArgumentParser()
ap.add_argument('dataset', type=str)
ap.add_argument('output', type=str)
args = ap.parse_args()


#################### train ####################

from _train.frame_interpolation.models.trainmodel import TrainModel
model = TrainModel()
# model = TrainModel().load_from_checkpoint \
#     ('temp/training_demo_output/checkpoints/epoch=0019-val_lpips=0.097225.ckpt')

# from _train.frame_interpolation.datasets.rrldextr import Datamodule
from _train.frame_interpolation.datasets.animerun import Datamodule
dm = Datamodule(args.dataset, bs=4)

trainer = pl.Trainer(
    precision=16,
    # fast_dev_run=True,
    # limit_train_batches=4,
    # limit_val_batches=4,
    # limit_test_batches=4,
    gradient_clip_val=1.0,
    max_epochs=20,

    default_root_dir=mkdir(args.output),

    accelerator='gpu',
    gpus=1,
    accumulate_grad_batches=8,

    callbacks=[pl.callbacks.ModelCheckpoint(
        monitor='val_lpips',
        mode='min',
        filename='{epoch:04d}-{val_lpips:0.6f}',
        save_top_k=8,
        dirpath=mkdir(f'{args.output}/checkpoints'),
        save_last=True,
    )],
    logger=[pl.loggers.TensorBoardLogger(
        mkdir(f'{args.output}/logs'),
        name='tensorboard',
        version=0,
        log_graph=False,
        default_hp_metric=True,
        prefix='',
    )],
)

trainer.fit(
    model,
    datamodule=dm,
)









