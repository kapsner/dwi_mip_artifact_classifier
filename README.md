# MRI DWI MIP artifact classifier

## Training

```bash
cd model
python model_execution.py -i /home/user/data/tech_artifacts/dwi_mips/pp_dwi_binary_tech_artifacts_221006/ \
    -e /home/user/development/trainings/dwi_mips/tech_artifacts_final_221006/ \
    -f param_config_dwi_ta_final_runs_2210.csv \
    -m training \
    -b 128 \
    -g 0 \
    -k 5
```

## Inference

```bash
python model_execution.py -i /home/user/data/tech_artifacts/dwi_mips/pp_dwi_binary_tech_artifacts_221006/ \
    -e /home/user/development/trainings/dwi_mips/tech_artifacts_final_221006/ \
    -m inference \
    -b 128 \
    -g 0 \
    -c /home/user/development/trainings/dwi_mips/tech_artifacts_final_221006/tensorboard/densenet121_adam_1472/fold_0/checkpoints/loss/valid=0.2644-epoch=192.ckpt \
    /home/user/development/trainings/dwi_mips/tech_artifacts_final_221006/tensorboard/densenet121_adam_1472/fold_1/checkpoints/loss/valid=0.3037-epoch=157.ckpt \
    /home/user/development/trainings/dwi_mips/tech_artifacts_final_221006/tensorboard/densenet121_adam_1472/fold_2/checkpoints/loss/valid=0.3386-epoch=196.ckpt \
    /home/user/development/trainings/dwi_mips/tech_artifacts_final_221006/tensorboard/densenet121_adam_1472/fold_3/checkpoints/loss/valid=0.2459-epoch=187.ckpt \
    /home/user/development/trainings/dwi_mips/tech_artifacts_final_221006/tensorboard/densenet121_adam_1472/fold_4/checkpoints/loss/valid=0.3163-epoch=138.ckpt
```
