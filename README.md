# RWKV-Unet
The codes for the work "Enhanced medical image segmentation using RWKV and CNN". I hope this will help you to reproduce the results.



## 2. Prepare data

- The datasets we used are provided by TransUnet's authors. [Get processed data in this link] (Synapse: https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd and ACDC: https://drive.google.com/drive/folders/1KQcrci7aKsYZi1hQoZ3T3QUtcy7b--n4).

## 3. Environment

- Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

## 4. Train/Test

- Run the train script on synapse dataset. The batch size we used is 4. Use larger batch size if you have enough memory

- Train

```bash
sh train.sh 
# or 
python train.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --root_path your DATA_DIR --max_epochs 150 --output_dir your OUT_DIR  --img_size 224 --base_lr 0.05 --batch_size 24
```

- Test 

```bash
sh test.sh 
# or 
python test.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path your DATA_DIR --output_dir your OUT_DIR --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24
```

## References and Acknowledgements
This code base uses certain code blocks and helper functions from:
* [TransUNet](https://github.com/Beckschen/TransUNet/tree/main)
* [Swin-Unet](https://github.com/Beckschen/TransUNet/tree/main) 
* [Vision-RWKV](https://github.com/OpenGVLab/Vision-RWKV)
* [CoTrFuse](https://github.com/BinYCn/CoTrFuse)


## Citation

```bibtex
@InProceedings{swinunet,
author = {Hu Cao and Yueyue Wang and Joy Chen and Dongsheng Jiang and Xiaopeng Zhang and Qi Tian and Manning Wang},
title = {Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation},
booktitle = {Proceedings of the European Conference on Computer Vision Workshops(ECCVW)},
year = {2022}
}

@misc{cao2021swinunet,
      title={Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation}, 
      author={Hu Cao and Yueyue Wang and Joy Chen and Dongsheng Jiang and Xiaopeng Zhang and Qi Tian and Manning Wang},
      year={2021},
      eprint={2105.05537},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
