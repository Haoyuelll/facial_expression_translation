### SPF_sad_2: 

- **loss:** loss_G_GAN * 1.0 + loss_vgg_perceptual * 1.0 + loss_masked_L1* 0.1

- **tmux:** face3

- **start_time:** 13:10, Dec 15

  ```python
  python train.py --dataroot ./datasets/motion_dataset --name SPF_sad_2 --CUT_mode CUT --gpu_ids 5 --display_id -1 --dataset_mode masked --lambda_NCE 0 --mask_start 17 --mask_end 68 --loss_mode 1
  ```

  

### SPF_sad_3: 

- **loss:** loss_G_GAN * 1.0 + loss_vgg_perceptual * 1.0 + loss_masked_L1* 0.1

- **tmux:** face4

- **start_time:** 21:23, Dec 6

  ```python
  python train.py --dataroot ./datasets/motion_dataset --name SPF_sad_3 --CUT_mode CUT --gpu_ids 2 --display_id -1 --dataset_mode masked --lambda_NCE 0 --loss_mode 1
  ```

  

### SPF_sad_4: 

- **loss:** loss_G_GAN * 1.0  + loss_NCE * 1.0 + loss_vgg_perceptual * 0.1

- **tmux:** face1

- **start_time:** 23:31, Dec 14

  ```python
  python train.py --dataroot ./datasets/motion_dataset --name SPF_sad_4 --CUT_mode CUT --gpu_ids 3 --display_id -1 --dataset_mode masked --lambda_VGG_perceptual 0.1 --lambda_L1_masked 0 --mask_start 17 --mask_end 68 --loss_mode 2
  ```

  

### SPF_sad_5: 

- **loss:** loss_G_GAN * 1.0 + loss_vgg_perceptual * 0.1  + loss_masked_L1 * 1.0

- **diff:** changing loss mode 2: calculate masked L1 loss of the background 

- **start_time:** 17:20, Dec 15

- **tmux:** face1

  ```python
  python train.py --dataroot ./datasets/motion_dataset --name SPF_sad_5 --CUT_mode CUT --gpu_ids 4 --display_id -1 --dataset_mode masked --lambda_VGG_perceptual 0.1 --lambda_L1_masked 1.0 --lambda_NCE 0 --mask_start 17 --mask_end 68 --loss_mode 2
  ```

  

### SPF_sad_6: 

- **loss:** loss_G_GAN * 1.0 + loss_vgg_perceptual * 1.0 + loss_masked_L1* 0.1

- **diff:** changing lambda

- **start_time:** 17:20, Dec 15

- **tmux:** face2

  ```python
  python train.py --dataroot ./datasets/motion_dataset --name SPF_sad_6 --CUT_mode CUT --gpu_ids 5 --display_id -1 --dataset_mode masked --lambda_NCE 0 --mask_start 17 --mask_end 68 --loss_mode 2
  ```

  

### SPF_sad_4: 

- **loss:** loss_G_GAN * 1.0 + loss_vgg_perceptual * 1.0 + loss_masked_L1* 1.0

- **diff:** changing lambda

- **start_time:** 17:20, Dec 15

- **tmux:** face4

  ```python
  python train.py --dataroot ./datasets/motion_dataset --name SPF_sad_7 --CUT_mode CUT --gpu_ids 6 --display_id -1 --dataset_mode masked --lambda_NCE 0 --lambda_L1_masked 1.0 --mask_start 17 --mask_end 68 --loss_mode 2
  ```

  

