source=${1:-M003}
reference=${2:-M009}
manipulator_ckpt=${3:-scmcl}
renderer_ckpt=${4:-M003}
manipulator_epoch=${5:-05}
renderer_epoch=${6:-01}
tag=${7:-scmcl}


python manipulator/test.py --celeb celeb/test/${actor}/ --checkpoints_dir exp/manipulator/${manipulator_ckpt} --which_epoch ${manipulator_epoch} --ref_dirs celeb/ref/${reference}/DECA --exp_name ref_on_${reference}_${tag}

./scripts/postprocess.sh celeb/test/${actor}/ ref_on_${reference}_${tag} exp/renderer/${renderer_ckpt} ${renderer_epoch}