poetry run python src/train.py --model_name cl-tohoku/bert-base-japanese-v2
poetry run python src/train.py --model_name cl-tohoku/bert-base-japanese-char-v2
poetry run python src/train.py --model_name cl-tohoku/bert-base-japanese
poetry run python src/train.py --model_name cl-tohoku/bert-base-japanese-char
poetry run python src/train.py --model_name cl-tohoku/bert-base-japanese-whole-word-masking
poetry run python src/train.py --model_name cl-tohoku/bert-large-japanese

poetry run python src/train.py --model_name studio-ousia/luke-japanese-base-lite
poetry run python src/train.py --model_name studio-ousia/luke-japanese-large-lite

poetry run python src/train.py --model_name bert-base-multilingual-cased
poetry run python src/train.py --model_name xlm-roberta-base
poetry run python src/train.py --model_name xlm-roberta-large
poetry run python src/train.py --model_name studio-ousia/mluke-base-lite
poetry run python src/train.py --model_name studio-ousia/mluke-large-lite

nohup poetry run python src/train.py --device "cuda:0" --model_name cl-tohoku/bert-base-japanese-v2 &
nohup poetry run python src/train.py --device "cuda:1" --model_name cl-tohoku/bert-base-japanese-char-v2 &
nohup poetry run python src/train.py --device "cuda:2" --model_name studio-ousia/luke-japanese-base-lite &
nohup poetry run python src/train.py --device "cuda:3" --model_name xlm-roberta-base &
