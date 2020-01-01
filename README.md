# Supervised Hashing with Latent Factor Model

## REQUIREMENTS
`pip install -r requirements.txt`

1. pytorch >= 1.0
2. loguru

## DATASETS
[cifar10-gist.mat](https://pan.baidu.com/s/1qE9KiAOTNs5ORn_WoDDwUg) password: umb6

[cifar-10_alexnet.t](https://pan.baidu.com/s/1ciJIYGCfS3m0marQvatNjQ) password: f1b7

[nus-wide-tc21_alexnet.t](https://pan.baidu.com/s/1YglFwoxB-3j7xTEyAc8ykw) password: vfeu

[imagenet-tc100_alexnet.t](https://pan.baidu.com/s/1ayv4wdtCOzEDsJy01SjRew) password: 6w5i

## USAGE
```
usage: run.py [-h] [--dataset DATASET] [--root ROOT]
              [--code-length CODE_LENGTH] [--num-samples NUM_SAMPLES]
              [--max-iter MAX_ITER] [--beta BETA] [--lamda LAMDA]
              [--topk TOPK]

LFH_PyTorch

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     Dataset name.
  --root ROOT           Path of dataset
  --code-length CODE_LENGTH
                        Binary hash code length.(default:
                        8,16,24,32,48,64,96,128)
  --num-samples NUM_SAMPLES
                        Number of samples.(default: 64)
  --max-iter MAX_ITER   Number of iterations.(default: 50)
  --beta BETA           Hyper-parameter.(default: 30)
  --lamda LAMDA         Hyper-parameter.(default: 1)
  --topk TOPK           Calculate top k data map.(default: all)
```

## EXPERIMENTS
cifar10-gist dataset. Gist features, 1000 query images, 5000 training images. beta = 30, lamda = 1.

cifar-10-alexnet dataset. Alexnet features, 1000 query images, 5000 training images. beta=1, lamda = 50.

nus-wide-tc21-alexnet dataset. Alexnet features, top 21 classes, 2100 query images, 10500 training images. beta = 1, lamda = 50.

imagenet-tc100-alexnet dataset. Alexnet features, top 100 classes, 5000 query images, 10000 training images. beta = 10, lamda = 40.

   Bits     | 8 | 16 | 24 | 32 | 48 | 64 | 96 | 128 
   ---        |   ---  |   ---   |   ---   |   ---   |   ---   |   ---   |   ---   |   ---   
  cifar10-gist@ALL  | 0.2339 | 0.2866  | 0.2968  | 0.3258  | 0.3339  | 0.3285  | 0.3419  | 0.3551
  cifar10-alexnet@ALL | 0.2994 | 0.3892 | 0.4032 | 0.4047 | 0.4302 | 0.4324 | 0.4365 | 0.4480
  nus-wide-tc21-alexnet@5000 | 0.6396 | 0.6615 | 0.6903 | 0.7131 | 0.7321 | 0.7423 | 0.7625 | 0.7619
  imagenet-tc100-alexnet@1000 | 0.0943 | 0.1870 | 0.2999 | 0.3738 | 0.4327 | 0.4754 | 0.5199 | 0.5344

