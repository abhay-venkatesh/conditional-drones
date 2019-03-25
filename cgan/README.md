# Conditional-GAN from pix2pix

## Training
```
python lib/model.py --mode train \
  --output_dir unreal_train \
  --max_epochs 200 \
  --input_dir dataset/unreal_proc/ \
  --which_direction AtoB

python lib/model.py --mode train\
  --output_dir icg_train \
  --max_epochs 200 \
  --input_dir dataset/icg_proc/ \
  --which_direction BtoA
```
