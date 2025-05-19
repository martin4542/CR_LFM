# python train_cloud_removal.py --exp SEN12_0515 \
#     --datadir /workspace/generative_model/data \
#     --batch_size 4 --num_epoch 500 \
#     --image_size 256 --f 8 --num_in_channels 4 --num_out_channels 4 \
#     --nf 256 --ch_mult 1 2 3 4 --attn_resolution 16 8 4 --num_res_blocks 2 \
#     --lr 1e-4 --scale_factor 1.0 --no_lr_decay \
#     --save_content --save_content_every 10 \
#     --use_grad_checkpointing


python train_cloud_removal.py --exp SEN12_0515 \
    --datadir /workspace/generative_model/data \
    --depth 12 \
    --batch_size 24 --num_epoch 500 \
    --image_size 256 --num_in_channels 4 \
    --lr 1e-5 --scale_factor 0.18215 --no_lr_decay \
    --save_content --save_content_every 10 