#python multi_train.py cifar100 -m resnet50 -c 7 --use-wandb -p project2 --cuda 2,3
#torchrun --nproc_per_node=2 --master_port=12345 multi_train.py cifar100 -m resnet34 -c 7 --use-wandb -p project2 --cuda 2,3 ; \
#torchrun --nproc_per_node=2 --master_port=12345 multi_train.py cifar100 -m resnet50 -c 7 --use-wandb -p project2 --cuda 2,3 ; \
#torchrun --nproc_per_node=2 --master_port=12345 multi_train.py cifar100 -m resnet101 -c 7 --use-wandb -p project2 --cuda 2,3 ; \
#torchrun --nproc_per_node=2 --master_port=12345 multi_train.py cifar100 -m resnet152 -c 7 --use-wandb -p project2 --cuda 2,3 ; \
#torchrun --nproc_per_node=2 --master_port=12345 multi_train.py cifar100 -m resnext50_16_8 -c 7 --use-wandb -p project2 --cuda 2,3 ; \
#torchrun --nproc_per_node=2 --master_port=12345 multi_train.py cifar100 -m resnext50_32_4 -c 7 --use-wandb -p project2 --cuda 2,3 ; \
#torchrun --nproc_per_node=2 --master_port=12345 multi_train.py cifar100 -m resnext101_32_4 -c 7 --use-wandb -p project2 --cuda 2,3 ; \
#torchrun --nproc_per_node=2 --master_port=12345 multi_train.py cifar100 -m resnext152_32_4 -c 7 --use-wandb -p project2 --cuda 2,3 ; \
#torchrun --nproc_per_node=2 --master_port=12345 multi_train.py cifar100 -m seresnet34 -c 7 --use-wandb -p project2 --cuda 2,3 ; \
#torchrun --nproc_per_node=2 --master_port=12345 multi_train.py cifar100 -m seresnet50 -c 7 --use-wandb -p project2 --cuda 2,3 ; \
#torchrun --nproc_per_node=2 --master_port=12345 multi_train.py cifar100 -m seresnet101 -c 7 --use-wandb -p project2 --cuda 2,3 ; \
#torchrun --nproc_per_node=2 --master_port=12345 multi_train.py cifar100 -m seresnet152 -c 7 --use-wandb -p project2 --cuda 2,3 ; \
#torchrun --nproc_per_node=2 --master_port=12345 multi_train.py cifar100 -m seresnext50_32_4 -c 7 --use-wandb -p project2 --cuda 2,3 ; \
#torchrun --nproc_per_node=2 --master_port=12345 multi_train.py cifar100 -m seresnext101_32_4 -c 7 --use-wandb -p project2 --cuda 2,3 ; \
#torchrun --nproc_per_node=2 --master_port=12345 multi_train.py cifar100 -m seresnext152_32_4 -c 7 --use-wandb -p project2 --cuda 2,3 ;
#torchrun --nproc_per_node=2 --master_port=12345 multi_train.py cifar100 -m resnext_cbam50_32_4 -c 7 --use-wandb -p project2 --cuda 2,3 ;
#torchrun --nproc_per_node=2 --master_port=12345 multi_train.py cifar100 -m resnext_cbam101_32_4 -c 7 --use-wandb -p project2 --cuda 2,3 ;
#torchrun --nproc_per_node=2 --master_port=12345 multi_train.py cifar100 -m efficientnet_v2_l -c 7 --use-wandb -p project2 --cuda 2,3 ;
#torchrun --nproc_per_node=2 --master_port=12345 multi_train.py cifar100 -m efficientnet_v2_m -c 7 --use-wandb -p project2 --cuda 2,3 ;
#torchrun --nproc_per_node=2 --master_port=12345 multi_train.py cifar100 -m efficientnet_v2_s -c 7 --use-wandb -p project2 --cuda 2,3 ;
#torchrun --nproc_per_node=2 --master_port=12345 multi_train.py cifar100 -m efficientnet_v2_xl -c 7 --use-wandb -p project2 --cuda 2,3 ;
#torchrun --nproc_per_node=2 --master_port=11234 multi_train.py cifar100 -m seresnext50_32_4_ver1 -c 7 --use-wandb -p project2 --cuda 0,1 ;
#torchrun --nproc_per_node=2 --master_port=11234 multi_train.py cifar100 -m seresnext50_32_4_ver2 -c 7 --use-wandb -p project2 --cuda 0,1 ;
#torchrun --nproc_per_node=2 --master_port=11234 multi_train.py cifar100 -m seresnext50_32_4_ver3 -c 7 --use-wandb -p project2 --cuda 0,1 ;
#torchrun --nproc_per_node=2 --master_port=11234 multi_train.py cifar100 -m seresnext50_32_4_ver4 -c 7 --use-wandb -p project2 --cuda 0,1 ;
#torchrun --nproc_per_node=2 --master_port=11234 multi_train.py cifar100 -m seresnext50_32_4_ver5 -c 7 --use-wandb -p project2 --cuda 0,1 ;
#torchrun --nproc_per_node=2 --master_port=11234 multi_train.py cifar100 -m seresnext50_32_4_ver6 -c 7 --use-wandb -p project2 --cuda 0,1 ;
#torchrun --nproc_per_node=2 --master_port=11234 multi_train.py cifar100 -m seresnext50_32_4_ver7 -c 7 --use-wandb -p project2 --cuda 0,1 ;
#torchrun --nproc_per_node=2 --master_port=11234 multi_train.py cifar100 -m seresnext50_32_4_ver8 -c 7 --use-wandb -p project2 --cuda 0,1 ;
#torchrun --nproc_per_node=2 --master_port=11234 multi_train.py cifar100 -m seresnext50_32_4_ver17 -c 7 --use-wandb -p project2 --cuda 0,1 ;
#torchrun --nproc_per_node=2 --master_port=11234 multi_train.py cifar100 -m seresnext50_32_4_ver18 -c 7 --use-wandb -p project2 --cuda 0,1 ;
#torchrun --nproc_per_node=2 --master_port=11234 multi_train.py cifar100 -m seresnext50_32_4_ver19 -c 7 --use-wandb -p project2 --cuda 0,1 ;
#torchrun --nproc_per_node=2 --master_port=11234 multi_train.py cifar100 -m seresnext50_32_4_ver20 -c 7 --use-wandb -p project2 --cuda 0,1 ;
#torchrun --nproc_per_node=2 --master_port=11234 multi_train.py cifar100 -m resnext29_32_4 -c 7 --use-wandb -p project2 --cuda 0,1 ;
#torchrun --nproc_per_node=2 --master_port=11234 multi_train.py cifar100 -m resnext29_16_4 -c 7 --use-wandb -p project2 --cuda 0,1 ;
#torchrun --nproc_per_node=2 --master_port=11234 multi_train.py cifar100 -m seresnext50_32_4_ver25 -c 7 --use-wandb -p project2 --cuda 0,1 ;
#torchrun --nproc_per_node=2 --master_port=11234 multi_train.py cifar100 -m seresnext50_32_4 --use-wandb -p project2 --cuda 8,9 -s -o "/home/oso0310/private/project2/task1";
torchrun --nproc_per_node=2 --master_port=11234 multi_train.py cifar100kd -m resnet34 --use-wandb -p project2 --cuda 8,9 -s -o "/home/oso0310/private/project2/task2";