actor=${1:-M003}


python renderer/train_scmcl.py \
    --celeb celeb/train \
    --train_root MEAD \
    --selected_actors ${actor} \
    --selected_actors_ref M003 M009 W029 M012 M030 W015 \
    --checkpoints_dir exp/renderer/${actor}_scmcl \
    --load_pretrain exp/renderer/${actor} \
    --manipulator_pretrain_weight exp/manipulator/scmcl/05_nets_finetuned.pth \
    --encoder_ckpt exp/scmc/image_finetune/final.pth \
    --dist_file exp/scmc/dists_for_image_mead/dists.pkl \
    --use_eyes_D \
    --n_frames_total 1 --batch_size 1 \
    --which_epoch 05 --niter 1 \
    --save_epoch_freq 1 --print_freq 10\
    --display_freq 5 --save_latest_freq 1000
