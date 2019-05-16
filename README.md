# Supervised Hashing with Latent Factor Model

论文[Supervised Hashing with Latent Factor Model](http://cs.nju.edu.cn/lwj/paper/SIGIR14_LFH.pdf)

## Requirements
1. pytorch 1.1
2. tb-nightly
3. tqdm
4. loguru

## 数据集下载地址
[cifar10-gist.mat](https://pan.baidu.com/s/1qE9KiAOTNs5ORn_WoDDwUg)

提取码: umb6

## 运行
`python run.py --dataset cifar10 --data-path <data_path> --code-length 64 `

日志记录在`logs`文件夹内

生成的hash code保存在`result`文件夹内，Tensor形式保存

## 参数说明
`dataset`: 使用数据集名称

`data-path`: 数据集文件路径

`code-length`: hash code长度

`num-samples`: 采样数量，（默认采样数量等于hash code length)

`max-iterations`: 迭代次数

`beta`: 超参，正态分布参数

`lamda`: 超参，Out-of-Sample Extension部分

`epsilon`： 控制迭代结束参数，代码里没有使用

## 实验
|bits|8|12|24|32|48|64|96|128|
|---|---|---|---|---|---|---|---|---|
|mAP|0.36|0.41|0.60|0.59|0.60|0.59|0.60|0.61|

上面给的数据没有多次取平均，不过我自己跑了很多次，结果都和上面相近
