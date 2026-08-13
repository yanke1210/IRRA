@echo off
cd /d "%~dp0"

echo Running CUHK-PEDES baseline...
python train.py --name iira_cuhk_pedes_a6 --img_aug --batch_size 64 --MLM --loss_names mlm+sdm --dataset_name CUHK-PEDES --root_dir D:\datasets --num_epoch 60

if errorlevel 1 (
    echo CUHK-PEDES training failed.
    exit /b 1
)

echo Running ICFG-PEDES baseline...
python train.py --name iira_icfg_pedes_a6 --img_aug --batch_size 64 --MLM --loss_names mlm+sdm --dataset_name ICFG-PEDES --root_dir D:\datasets --num_epoch 60

if errorlevel 1 (
    echo ICFG-PEDES training failed.
    exit /b 1
)

echo Running RSTPReid baseline...
python train.py --name iira_rstpreid_a6 --img_aug --batch_size 64  --MLM --loss_names mlm+sdm --dataset_name RSTPReid --root_dir D:\datasets --num_epoch 60

if errorlevel 1 (
    echo RSTPReid training failed.
    exit /b 1
)

echo All trainings completed.
