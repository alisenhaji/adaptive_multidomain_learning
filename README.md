

##### Requirements
- Pytorch (at least version 3.0)
- COCO API (from https://github.com/cocodataset/cocoapi)

##### Launching the code
First download the data with ``download_data.sh /path/to/save/data/``. Please copy ``decathlon_mean_std.pickle`` to the data folder. 

To train a dataset with parallel adapters put on a pretrained 'off the shelf' deep network:

``CUDA_VISIBLE_DEVICES=2 python train_new_task_adapters.py --dataset cifar100 --wd1x1 1. --wd 5. --mode parallel_adapters --source /path/to/net``

##### Pretrained network
Pretrained network on ImageNet (with reduced resolution):
- a ResNet 26 inspired from the original ResNet from [He,16]: https://drive.google.com/open?id=1y7gz_9KfjY8O4Ue3yHE7SpwA90Ua1mbR
