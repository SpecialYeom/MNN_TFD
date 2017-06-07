for i in 0.001 0.01 0.05 0.1 0.5 1 
do
  python train.py --lr $i
done
