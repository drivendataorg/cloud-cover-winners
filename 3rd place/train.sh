mkdir -p logs

echo "Training..."
python -m torch.distributed.launch --nproc_per_node=4 train.py --fold=2 --bands=4 --epoches=100 --encoder=tf_efficientnetv2_s --checkpoint=tf_efficientnetv2_s_4b 2>&1 | tee logs/out_tf_efficientnetv2_s_4b_2.txt
python -m torch.distributed.launch --nproc_per_node=4 train.py --fold=2 --bands=4 --minmax=True --epoches=90 --encoder=tf_efficientnetv2_b0 --checkpoint=tf_efficientnetv2_b0_4b_minmax 2>&1 | tee logs/out_tf_efficientnetv2_b0_4b_minmax_2.txt
python -m torch.distributed.launch --nproc_per_node=4 train.py --fold=2 --bands=3 --epoches=80 --encoder=tf_efficientnetv2_b0 --checkpoint=tf_efficientnetv2_b0_3b 2>&1 | tee logs/out_tf_efficientnetv2_b0_3b_2.txt
python -m torch.distributed.launch --nproc_per_node=4 train.py --fold=2 --bands=3 --epoches=60 --encoder=resnet34 --checkpoint=res34_3b_scaled 2>&1 | tee logs/out_res34_3b_scaled_2.txt
python -m torch.distributed.launch --nproc_per_node=4 train.py --fold=2 --bands=3 --minmax=True --epoches=60 --encoder=resnet34 --checkpoint=res34_3b_minmax 2>&1 | tee logs/out_res34_3b_minmax_2.txt
python -m torch.distributed.launch --nproc_per_node=4 train.py --fold=2 --bands=4 --epoches=60 --norm_max=20000 --encoder=resnet34 --checkpoint=res34_pretrained 2>&1 | tee logs/out_res34_pretrained_2.txt
python -m torch.distributed.launch --nproc_per_node=4 train.py --fold=2 --bands=4 --epoches=70 --encoder=tf_efficientnet_b3_ns --checkpoint=b3_4bands 2>&1 | tee logs/out_b3_4bands_2.txt
python -m torch.distributed.launch --nproc_per_node=4 train.py --fold=2 --bands=4 --epoches=60 --norm_max=30000 --encoder=tf_efficientnet_b0_ns --checkpoint=b0_4bands 2>&1 | tee logs/out_tf_b0_4bands_2.txt

echo "All models trained!"