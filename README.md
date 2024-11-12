# Hourglass Perception Matters
A lightweight Stacked Hourglass Network for Human Pose Estimation [arxiv](https://arxiv.org/pdf/2302.04815)

Code based on Pytorch implementation of Stacked Hourglass paper [here](https://github.com/anibali/pytorch-stacked-hourglass)

Environment setup

``` 
conda env create -f environment.yml
conda activate hourglass 
pip install -r requirements.txt
```

For training and evaluation
Set the path to images and checkpoint in ```train.sh``` and ```eval.sh```

The model is setup using ```config.py```. The best model is as described in poster below. 

![Poster](<https://github.com/jameelhassan/PoseEstimation/blob/main/final checkpoint/poster.png?raw=true>)


