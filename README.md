## About

This is a simple implementation of event camera image reconstruction, just for fun. 

Reference: [CVPR19 - Event-Based High Dynamic Range Image and Very High Frame Rate Video Generation Using Conditional Generative Adversarial Networks](https://arxiv.org/pdf/1811.08230.pdf). 

Detailed explanation of model is in the paper (only available via academic request). Our model only uses U-Net and residual blocks.

## Requirements

Before running the code, please check your package versions by

```shell
pip3 install -r requirements.txt
```

After that, you should run

```shell
python3 get_uzh_dataset.py
```

 to get data generated by DAVIS240C. The downloaded data will be saved in a new  `dataset` directory. You can also download the dataset by your own and unzip it in an empty directory `dataset`. Link: [UZH Event Camera Dataset](http://rpg.ifi.uzh.ch/davis_data.html). 

Then, for example, you can run

```shell
python3 main.py --lr 0.001 --train_batch 32 --test_batch 16 -c 1 --fixed --resume
```

to resume training model from checkpoint (if you had already generated checkpoint) with 1-channel event peseudo image with fixed value.

## Experiments

Below is our experiment results.

|    Data   |   SSIM  |   PSNR   |
| :-------: | :-----: | :------: |
|    1ch    | 0.78505 | 24.95658 |
| 1ch_fixed | 0.73154 | 23.27854 |
|    2ch    | 0.79131 | 25.38525 |
| 2ch_fixed | 0.77254 | 24.66690 |
|    3ch    | 0.81217 | 23.60319 |
| 3ch_fixed | 0.75738 | 24.33863 |
