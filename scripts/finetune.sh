AZFUSE_USE_FUSE=0 NCCL_ASYNC_ERROR_HANDLING=0 python finetune_sdm_yaml.py --cf config/ref_attn_clip_combine_controlnet/tiktok_S256L16_xformers_tsv.py \
--do_train --root_dir /home1/wangtan/code/ms_internship2/github_repo/run_test \ 
--local_train_batch_size 32 \
--local_eval_batch_size 32 \
--log_dir exp/tiktok_ft \ 
--epochs 20 --deepspeed \
--eval_step 500 --save_step 500 \
--gradient_accumulate_steps 1 \
--learning_rate 2e-4 --fix_dist_seed --loss_target "noise" \
--train_yaml /home/wangtan/data/disco/yaml_file/train_TiktokDance-poses-masks.yaml \
--val_yaml /home/wangtan/data/disco/yaml_file/new10val_TiktokDance-poses-masks.yaml \
--unet_unfreeze_type "all" \
--refer_sdvae \
--ref_null_caption False \
--combine_clip_local --combine_use_mask \
--conds "poses" "masks" \
--stage1_pretrain_path /path/to/pretrained_model_checkpoint/mp_rank_00_model_states.pt 