PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python AZFUSE_USE_FUSE=0 NCCL_ASYNC_ERROR_HANDLING=0 \
python finetune_sdm_yaml.py \
--cf config/ref_attn_clip_combine_controlnet_attr_pretraining/coco_S256_xformers_tsv_strongrand7.py \
--eval_visu --root_dir /home/nfs/jsh/DisCo \
--local_train_batch_size 1 \
--local_eval_batch_size 1 \
--log_dir exp/eval --epochs 20 --deepspeed --eval_step 500 --save_step 500 \
--gradient_accumulate_steps 1 \
--learning_rate 2e-4 --fix_dist_seed --loss_target "noise" \
--train_yaml /home/nfs/jsh/HOME/VITON-hd-resized/try2.0/tsv/val.yaml \
--val_yaml /home/nfs/jsh/HOME/VITON-hd-resized/try2.0/tsv/val.yaml \
--unet_unfreeze_type "all" \
--ref_null_caption False \
--combine_clip_local --combine_use_mask --viton_hd --no_smpl --use_cf_attn \
--stage1_pretrain_path /home/nfs/jsh/DisCo/exp/pretrain_3.0_1_dino_hd/22499.pth/mp_rank_00_model_states.pt \
--eval_save_filename /home/nfs/jsh/DisCo/eval/eval_pt3.0_1_dino_hd_try4.0_fix_fat_vedio3 