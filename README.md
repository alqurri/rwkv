# RWKVFuse
The codes for the work "Enhanced medical image segmentation using RWKV and CNN". I hope this will help you to reproduce the results.



## 2. Prepare data

- The datasets we used are provided by TransUnet's authors. [Get processed data in this link] (Synapse: https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd and ACDC: https://drive.google.com/drive/folders/1KQcrci7aKsYZi1hQoZ3T3QUtcy7b--n4).

## 3. Environment

- Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.
- Download [Vision-RWKV](https://github.com/OpenGVLab/Vision-RWKV) and place ./vrwkv/cuda/wkv_op.cpp and ./vrwkv/cuda/wkv_cuda.cu in the same directory.

## 4. Train/Test

- Run the train script on synapse dataset. 

- Train

```bash

python train.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --root_path  './project_TransUNet/data/Synapse/train_npz' --list_dir  './project_TransUNet/TransUNet/lists/lists_Synapse' --n_class  9 --max_epochs 150 --output_dir 'Your ouput dir'  --img_size 224 --base_lr 0.05 --batch_size 24
```

- Test 

```bash
 
python test.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path your DATA_DIR --output_dir your OUT_DIR --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24
```

## References and Acknowledgements
This code base uses certain code blocks and helper functions from:
* [TransUNet](https://github.com/Beckschen/TransUNet/tree/main)
* [Swin-Unet](https://github.com/Beckschen/TransUNet/tree/main) 
* [Vision-RWKV](https://github.com/OpenGVLab/Vision-RWKV)
* [CoTrFuse](https://github.com/BinYCn/CoTrFuse)


## Citation


