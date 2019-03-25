# conditional-drones
Using Conditional GANS for generating infinite data to train drone vision models.  

## Repository organization
```
docs/
cgan/
  lib/
    model.py
    utils.py
    ...
  main.py
data/
  unreal/
  icg/
```

## Training cgan
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
