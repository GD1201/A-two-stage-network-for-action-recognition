# A-two-stage-network-for-action-recognition
action recognition
model includes STGAIN and FJMAE;
data is our own self-constructed human-robot interaction dataset.
Img is model architechture.
####NOTICE 


Train STGAIN
cd model/STAGIN 
python main.py --data_name skeleton --batch_size 24 --iterations 1000

Train FJMAR

cd model/FJMAE 

pretrained stage : 
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
#python -m torch.distributed.run --nproc_per_node=1 --master_port 8203 main_pretrain.py \
python main_pretrain.py
--batch_size 32 \
--dataset ntu-xsub/ntu-view/18-1 \
--model mae_vit_base_patch16 \
--weight_decay 0.05 \
--frame_ratio 0.5 \
--joint_ratio 0.5 \
--num_workers 10  \
--output_dir  /home/huaizhenhao/codes/GD/mae/x-sub-0.5-0.5-8 \
--log_dir  /home/huaizhenhao/codes/GD/mae/x-sub-0.5-0.5-8  \
--epochs  800 \
--warmup_epochs 20 \
--norm_pix_loss \
--lr 3e-4  \
--use_mask_tokens


finetune stage :
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
python main_finetune.py
--batch_size 24 \
--model vit_base_patch16  \
--weight_decay 0.05 \
--num_workers 10  \
--output_dir /\
--log_dir \
--epochs 100 \
--warmup_epochs 5 \
--lr 3e-4  \
--finetune  /home/huaizhenhao/codes/GD/mae/output/x-sub-0.5-0.5-7/checkpoint-799.pth \
--min_lr 1e-5





