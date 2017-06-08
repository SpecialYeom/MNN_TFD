for batch_size in 10 100 500
do
  python3 train.py --batch-size $batch_size --lr 0.1 --momentum 0
done
