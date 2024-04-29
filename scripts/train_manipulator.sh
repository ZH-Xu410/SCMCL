python manipulator/train_scmcl.py \
    --train_root MEAD \
    --selected_actors M003 M009 W029 M012 M030 W015 \
    --checkpoints_dir exp/manipulator/scmcl \
    --dist_file exp/scmc/dists_for_3dmm/dists.pkl \
    --encoder_ckpt exp/scmc/3dmm_finetune/final.pth \
    --finetune --niter 5 --niter_decay 5 \
    --print_freq 50