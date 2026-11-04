# test
python train.py --name iira_rs_test --img_aug --batch_size 100  --MLM --loss_names mlm+sdm+tc --dataset_name RSICD --root_dir /home/coco/datasets --sampler identity


python test.py --config_file logs/RSICD/20261103_105444_iira_rs_test/configs.yaml



# train时 evaluate 使用 val split数据（改的 config yaml）

python train.py --name iira_rs --img_aug --batch_size 100  --MLM --loss_names mlm+sdm+tc --dataset_name RSICD --root_dir /home/coco/datasets --sampler identity


python test.py --config_file logs/RSICD/20261103_133307_iira_rs/configs.yaml


# sampler 改为 random

python train.py --name iira_rs_random --img_aug --batch_size 100  --MLM --loss_names mlm+sdm+tc --dataset_name RSICD --root_dir /home/coco/datasets --sampler random


python test.py --config_file logs/RSICD/20261104_090525_iira_rs_random/configs.yaml


# v1 版本： tc与 sdm解耦

python train.py --name iira_rs_v1 --img_aug --batch_size 100  --MLM --loss_names mlm+sdm+tc --dataset_name RSICD --root_dir /home/coco/datasets --sampler identity


python test.py --config_file logs/RSICD/20261104_110201_iira_rs_v1/configs.yaml

# 效果比 20261103_133307_iira_rs 还差  0.3

# 要不要改成random试一下？？？？？？？？？？？？


# v2：修改 gap 不是整个gallery上的置信度与正确率之差，而是 样本对之间的

python train.py --name iira_rs_v1 --img_aug --batch_size 100  --MLM --loss_names mlm+sdm+tc --dataset_name RSICD --root_dir /home/coco/datasets --sampler identity


python test.py --config_file logs/RSICD/20261104_110201_iira_rs_v2/configs.yaml



# 要不要改成random试一下？？？？？？？？？？？？


# v3 gamma更新规则不对
