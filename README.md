# Supervised Hashing with Latent Factor Model

## REQUIREMENTS
`pip install -r requirements.txt`

## DATASETS
[cifar10-gist.mat](https://pan.baidu.com/s/1qE9KiAOTNs5ORn_WoDDwUg)

password: umb6

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
cifar10-gist dataset. 1000 query images, 59000 training images.

 | | 8 bits | 16 bits | 24 bits | 32 bits | 48 bits | 64 bits | 96 bits | 128 bits
   :-:   |  :-:    |   :-:   |   :-:   |   :-:   |   :-:   |   :-:   |   :-:   
cifar10-gist MAP@ALL | 0.4662 | 0.5591 | 0.5935 | 0.5912 | 0.6117 | 0.6073 | 0.6210 | 0.6321
