@echo off
cd /d "%~dp0"

echo Running CUHK-PEDES KER+IRR...
python train.py --name ker_irr_cuhk_pedes --img_aug --batch_size 64 --MLM --loss_names ker+irr --dataset_name CUHK-PEDES --root_dir D:\datasets --num_epoch 60

if errorlevel 1 (
    echo CUHK-PEDES training failed.
    exit /b 1
)

echo Running ICFG-PEDES KER+IRR...
python train.py --name ker_irr_icfg_pedes --img_aug --batch_size 64 --MLM --loss_names ker+irr --dataset_name ICFG-PEDES --root_dir D:\datasets --num_epoch 60

if errorlevel 1 (
    echo ICFG-PEDES training failed.
    exit /b 1
)

echo Running RSTPReid KER+IRR...
python train.py --name ker_irr_rstpreid --img_aug --batch_size 64  --MLM --loss_names ker+irr --dataset_name RSTPReid --root_dir D:\datasets --num_epoch 60

if errorlevel 1 (
    echo RSTPReid training failed.
    exit /b 1
)

echo All trainings completed.
