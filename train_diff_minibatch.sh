for i in 1 10 500 1000
do
  python train.py --batch-size $i
done
