python -m prepocess.build_dicts
python -m prepocess.embd
python -m prepocess.user_embd
python -m prepocess.build_train --processes=10
python -m prepocess.build_dev --processes=10
python -m prepocess.build_test --processes=20