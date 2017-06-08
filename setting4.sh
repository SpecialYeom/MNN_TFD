for batch_size in 10 100 500
do
  python3 train.py --batch-size $batch_size --lr 0.01 --momentum 0 --gpu-id 1
done
