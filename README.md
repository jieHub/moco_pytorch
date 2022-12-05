# moco_pytorch

- [official pytorch implementation](https://github.com/facebookresearch/moco)

## DataSet && Preprocess

- [kaggle tiny-imagenet](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet)
```bash
unzip archive.zip
mkdir data && cd data && ls ../tiny-imagenet-200/train | while read name ; do mkdir -p ./${name}; cp ../tiny-imagenet-200/train/${name}/images/* ./${name}; done
```

## train
```
python train.py -a resnet50 --lr 0.015 --batch-size 128 --dist-url 'tcp://127.0.0.1:6772' --multiprocessing-distributed --world-size 1 --rank 0 ./data
```

