PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python AZFUSE_USE_FUSE=0 QD_USE_LINEIDX_8B=0 NCCL_ASYNC_ERROR_HANDLING=0 python finetune_sdm_yaml.py \
--cf config/ref_attn_clip_combine_controlnet_attr_pretraining/coco_S256_xformers_tsv_strongrand1.py \
--do_train --root_dir /home/nfs/jsh/DisCo \
--local_train_batch_size 64 --local_eval_batch_size 64 --log_dir exp/pretrain1gpu \
--epochs 100 --deepspeed --eval_step 200 --save_step 200 --gradient_accumulate_steps 1 \
--learning_rate 5e-4 --fix_dist_seed --loss_target "noise" \
--train_yaml /HOME/HOME/jisihui/VITON-hd-resized/train/tsv/train.yaml \
--val_yaml /HOME/HOME/jisihui/VITON-hd-resized/try/tsv/val.yaml \
--unet_unfreeze_type "transblocks"  --ref_null_caption False \
--combine_clip_local --combine_use_mask --viton \
--conds "masks" --max_eval_samples 2000 --strong_aug_stage1 --node_split_sampler 0 \
--resume