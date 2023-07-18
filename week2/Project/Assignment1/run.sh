#torchrun --nproc_per_node=2 --master_port=11345 main.py --gpu_id 2,3 --batch_size 64 --test_batch_size 16 --epochs 30
#python main.py --gpu_id 7,9 --batch_size 64 --test_batch_size 16 --epochs 300
#python main.py --gpu_id 2 --batch_size 64 --test_batch_size 16 --epochs 30
torchrun --nproc_per_node=2 --master_port=12345 main.py --gpu_id 2,3 --batch_size 64 --test_batch_size 16 --epochs 30