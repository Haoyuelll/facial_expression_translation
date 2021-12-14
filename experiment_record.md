### SPF_sad_2: 

- **loss:** loss_G_GAN * 1.0 + loss_vgg_perceptual * 1.0 + loss_masked_L1* 0.1

- **tmux:** face1

- **start_time:** 21:22, Dec 6

  ```python
  python train.py --dataroot ./datasets/motion_dataset --name SPF_sad_2 --CUT_mode CUT --gpu_ids 5 --display_id -1 --dataset_mode masked --lambda_NCE 0 --mask_start 17 --mask_end 68 --loss_mode 1
  ```

  

### SPF_sad_3: 

- **loss:** loss_G_GAN * 1.0 + loss_vgg_perceptual * 1.0 + loss_masked_L1* 0.1

- **tmux:** face2

- **start_time:** 21:23, Dec 6

  ```python
  python train.py --dataroot ./datasets/motion_dataset --name SPF_sad_3 --CUT_mode CUT --gpu_ids 2 --display_id -1 --dataset_mode masked --lambda_NCE 0  --loss_mode 1
  ```

  

### SPF_sad_4: 

- **loss:** loss_G_GAN * 1.0  + loss_NCE * 1.0 + loss_vgg_perceptual * 0.1

- **tmux:** face1

- **start_time:** 0:49, Dec 11

  ```python
  python train.py --dataroot ./datasets/motion_dataset --name SPF_sad_4 --CUT_mode CUT --gpu_ids 3 --display_id -1 --dataset_mode masked --lambda_VGG_perceptual 0.1 --lambda_L1_masked 0 --mask_start 17 --mask_end 68 --loss_mode 2
  ```

  

### SPF_sad_5: 

- **loss:** loss_G_GAN * 1.0  + loss_NCE * 1.0 + loss_vgg_perceptual * 0.1

- **tmux:** face2

- **start_time:** 0:49, Dec 11

  ```python
  python train.py --dataroot ./datasets/motion_dataset --name SPF_sad_5 --CUT_mode CUT --gpu_ids 4 --display_id -1 --dataset_mode masked --lambda_VGG_perceptual 0.1 --lambda_L1_masked 0 --mask_start 17 --mask_end 68 --loss_mode 2
  ```

  

### SPF_sad_3: 

- **loss:** loss_G_GAN * 1.0 + loss_vgg_perceptual * 1.0 + loss_masked_L1* 0.1

- **tmux:** face2

- **start_time:** 21:23, Dec 6

  ```python
  python train.py --dataroot ./datasets/motion_dataset --name SPF_sad_3 --CUT_mode CUT --gpu_ids 2 --display_id -1 --dataset_mode masked --lambda_NCE 0  --loss_mode 1
  ```

  

### SPF_sad_4: 

- **loss:** loss_G_GAN * 1.0  + loss_NCE * 1.0 + loss_vgg_perceptual * 0.1

- **tmux:** face3

- **start_time:** 11:27, Dec 6

  ```python
  python train.py --dataroot ./datasets/motion_dataset --name SPF_sad_4 --CUT_mode CUT --gpu_ids 3 --display_id -1 --dataset_mode masked --lambda_VGG_perceptual 0.1 --lambda_L1_masked 0 --mask_start 17 --mask_end 68 --loss_mode 2
  ```

  

