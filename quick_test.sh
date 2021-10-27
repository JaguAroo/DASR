# super-resolve an LR image (x2) using the model trained on noise-free degradations with isotropic Gaussian blurs
python quick_test.py --img_dir='./dataset/X2/0001x2.png' \
                     --scale='2' \
                     --resume=600 \
                     --blur_type='iso_gaussian'

# super-resolve an LR image (x4) using the model trained on general degradations with anisotropic Gaussian blurs and noises
# python quick_test.py --img_dir='D:/CODE/DASRREPO/DATA/DIV2K_train_LR_bicubic/X4/0001x4.png' \
#                      --scale='4' \
#                      --resume=600 \
#                      --blur_type='aniso_gaussian'

cmd /k
