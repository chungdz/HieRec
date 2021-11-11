python -m prepocess.build_dicts
python -m prepocess.embd
python -m prepocess.user_embd
python -m prepocess.build_train --processes=10
python -m prepocess.build_dev --processes=10
CUDA_VISIBLE_DEVICES=0,1,2,3 python training.py --gpus=4 --epoch=10
CUDA_VISIBLE_DEVICES=0,1,2,3 python validate.py --gpus=4 --epoch=10 --filenum=10