# Supervised Contrastive Learning with Hard Negative Samples

This is the official code for the paper "Supervised Contrastive Learning with Hard Negative Samples". This repository contains the implementation of HSCL and related experiments described in the paper.

## Implenment on image dataset
The default set up is cifar100 using H-SCL with beta = 5, The accuracy with KNN classifier will be saved as npy file.

```
python main.py
```
The learned model will be saved in folder "result", please use linear.py to show the result with linear classifier.

For reproduce the loss shown in Figure 7, please use command
```
python main_loss.py
```
Value of four different loss will be saved as npy file.

## Citation

If you find this repo useful for your research, please consider citing the paper:

```
@article{jiang2022supervised,
  title={Supervised Contrastive Learning with Hard Negative Samples},
  author={Jiang, Ruijie and Nguyen, Thuan and Ishwar, Prakash and Aeron, Shuchin},
  journal={arXiv preprint arXiv:2209.00078},
  year={2022}
}
```
For any questions, please contact Ruijie Jiang (Ruijie.Jiang@tufts.edu)

## Acknowledgements
This code is a modified version of the HCL implementation by [Josh/HCL](https://github.com/joshr17/HCL). 
Part of this code is inspired by [leftthomas/SimCLR](https://github.com/leftthomas/SimCLR) and by [Josh/HCL](https://github.com/joshr17/HCL).
