# test
python train.py --name iira_rs_test --img_aug --batch_size 100  --MLM --loss_names mlm+sdm+ot --dataset_name RSICD --root_dir /home/coco/datasets --sampler identity


python test.py --config_file logs/RSICD/20261103_105444_iira_rs_test/configs.yaml



# train时 evaluate 使用 val split数据（改的 config yaml）

python train.py --name iira_rs --img_aug --batch_size 100  --MLM --loss_names mlm+sdm+ot --dataset_name RSICD --root_dir /home/coco/datasets --sampler identity


python test.py --config_file logs/RSICD/20261103_133307_iira_rs/configs.yaml


# sampler 改为 random

python train.py --name iira_rs_random --img_aug --batch_size 100  --MLM --loss_names mlm+sdm+ot --dataset_name RSICD --root_dir /home/coco/datasets --sampler random


python test.py --config_file logs/RSICD/20261104_090525_iira_rs_random/configs.yaml

