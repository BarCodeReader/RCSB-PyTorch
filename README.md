## Recursive Contour Saliency Blending Network for Accurate Salient Object Detection
<img src="https://github.com/BarCodeReader/RCSB-PyTorch/blob/main/bin/WACV_LOGO.jpeg" alt="drawing" width="500"/>

### Network Architecture
![network](https://github.com/BarCodeReader/RCSB-PyTorch/blob/main/bin/RCSBNet.png)
### Prerequisites
Ubuntu 18.04\
Python==3.8.3\
Torch==1.8.0+cu111\
Torchvision=0.9.0+cu111\
Kornia

### Dataset
For all datasets, they should be organized in below's fashion:
```
|__dataset_name
   |__Images: xxx.jpg ... 
   |__Masks : xxx.png ... 
```
For training, put your dataset folder under:
```
dataset/
```
For evaluation, download below datasets and place them under:
```
dataset/benchmark/
```
- [ECSSD](http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)
- [HKU-IS](https://i.cs.hku.hk/~gbli/deep_saliency.html)
- [DUTS-TE](http://saliencydetection.net/duts/)
- [DUT-OMRON](http://saliencydetection.net/dut-omron/)
- [PASCAL-S](http://cbi.gatech.edu/salobj/)
### Train & Test
**Firstly, make sure you have enough GPU RAM**.\
With default setting (batchsize=4), 24GB RAM is required, but you can always reduce the batchsize to fit your hardware.

Default values in option.py are already set to the same configuration as our paper, so\
to train the model, simply:
```
python main.py --GPU_ID 0
```
to test the model, simply:
```
python main.py --test_only --pretrain "bal_bla.pt" --GPU_ID 0
```
If you want to train/test with different settings, please refer to **option.py** for more control options.\
Currently only support training on single GPU.
### Pretrain Model & Pre-calculated Saliency Map
Our pretrain model and pre-calculated saliency map: [[Google]](https://drive.google.com/drive/folders/1P5h-L_YhbDls98r0NWXGMOcG6tjZNjza?usp=sharing)

If you have problem loading the model due to latest torch use zip file as serialization, download the "RCSB_old_style.pt" instead. It is the same as "RCSB.pt", just to fit older torch versions.

### Evaluation
Firstly, obtain predictions via
```
python main.py --test_only --pretrain "bal_bla.pt" --GPU_ID 0 --save_result
```
Output will be saved in `./output/` by default.

For *PR curve* and *F curve*, we use the code provided by this repo: [[BASNet, CVPR-2019]](https://github.com/xuebinqin/Binary-Segmentation-Evaluation-Tool)\
For *MAE*, *F measure*, *E score* and *S score*, we use the code provided by this repo: [[F3Net, AAAI-2020]](https://github.com/weijun88/F3Net#evaluation)
### Evaluation Results
#### Qualitative Results
![pred](https://github.com/BarCodeReader/RCSB-PyTorch/blob/main/bin/vis.png)
![contour](https://github.com/BarCodeReader/RCSB-PyTorch/blob/main/bin/ctr.png)
#### Quantitative Results
![mae_table](https://github.com/BarCodeReader/RCSB-PyTorch/blob/main/bin/MAE_table.png)
![prfm_curve](https://github.com/BarCodeReader/RCSB-PyTorch/blob/main/bin/all_prfm_curves_12.png)
### Citation
If you like this work, please cite our paper
```
@misc{yun2021recursive,
      title={Recursive Contour Saliency Blending Network for Accurate Salient Object Detection},
      author={Yi Ke Yun and Chun Wei Tan and Takahiro Tsubono},
      year={2021},
      eprint={2105.13865},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
