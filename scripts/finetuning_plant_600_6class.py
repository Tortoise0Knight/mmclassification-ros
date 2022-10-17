_base_ = [
    '../_base_/models/resnet50.py',
    '../_base_/default_runtime.py'
]

# Model config
model = dict(
    backbone=dict(
        frozen_stages=3,   #冻结前三层
        init_cfg=dict(
            type='Pretrained',
            checkpoint='D:\\mmclassification\\models\\resnet50_8xb32_in1k_20210831-ea4938fc.pth',
            prefix='backbone',
        )),
    head=dict(num_classes=6),
)

# Dataset config
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],  #??????????
    std=[58.395, 57.12, 57.375], 
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

dataset_type = 'CustomDataset'
classes = ['V', 'C', 'H', 'L', 'M', 'W']  # The category names of your dataset

data = dict(
    samples_per_gpu=32, #????
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='D:\\mmclassification\\data\\plant_dataset\\train',
        ann_file='D:\\mmclassification\\data\\plant_dataset\\meta\\train.txt',
        classes=classes,
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        data_prefix='D:\\mmclassification\\data\\plant_dataset\\val',
        ann_file='D:\\mmclassification\\data\\plant_dataset\\meta\\val.txt',
        classes=classes,
        pipeline=test_pipeline
    ),
    test=dict(
        type=dataset_type,
        data_prefix='D:\\mmclassification\\data\\plant_dataset\\test',
        ann_file='D:\\mmclassification\\data\\plant_dataset\\meta\\test.txt',
        classes=classes,
        pipeline=test_pipeline
    )
)
evaluation = dict(interval=10, save_best=True, metric='accuracy', metric_options={'topk': 1}) #?????

# Training schedule config
# lr is set for a batch size of 128
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=[15])
runner = dict(type='EpochBasedRunner', max_epochs=100)
log_config = dict(interval=10)