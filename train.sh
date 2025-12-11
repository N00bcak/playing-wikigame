xport LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
export LD_LIBRARY_PATH=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"):$LD_LIBRARY_PATH
export NCCL_CUMEM_ENABLE=0
export LP_DEBUG=1
export LP_LOG_LEVEL=DEBUG
#export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export CUDA_VISIBLE_DEVICES=0
export NCCL_DEBUG=INFO

N_GPUS=8
GRADIENT_BATCH_SIZE=128

# Note we kind of cheat and use Qwen3-1.7B which is already instruct-tuned.
yes n | python train.py \
    --env_id game:WikiGame-v0-hard \
    --wrappers concat \
    --prompt_template qwen3_game \
    --gamma 0.9 \
    --gpus $N_GPUS \
    --gradient-checkpointing \
    --num_samples 1 \
    --rollout_batch_size $GRADIENT_BATCH_SIZE \
    --num_env 1 \
    --rollout_batch_size_per_device $((GRADIENT_BATCH_SIZE / N_GPUS)) \
    --pi_buffer_maxlen_per_device $((GRADIENT_BATCH_SIZE / N_GPUS)) \
    --pretrain Qwen/Qwen3-1.7B \
    --enable_prefix_caching \
    --vllm_gpu_ratio 0.25 \
    --vllm_sleep \
    --collocate \
    --rnd-seed \
    --learning_rate 0.000001 \
    --lr_scheduler constant \
    --lr_warmup_ratio 0 \
    --num_ppo_epochs 2 \
    --train_batch_size $GRADIENT_BATCH_SIZE \
    --train_batch_size_per_device 1 \
    --beta 0 \
    --max_model_len 32000 \
    --generate_max_length 4096 \
    --temperature 1 \
    --top_p 1 \
    --eval_steps -1 \
    --save_steps -1 \
    --eval_temperature 0.6 \
    --eval_top_p 0.95 \
    --eval_generate_max_length 1024 \
    --max_train 65536 \
    --max_save_num 8 \
    --use-wb \
    --wb-run-name gem-wikigame-instruct-cluster \
    --save_path gem-wikigame-instruct-cluster \
    --wb_project gem