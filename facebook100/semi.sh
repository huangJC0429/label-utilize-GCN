
----facebook100----
python test-flan.py --dataset Caltech36 --epochs 2000 --hidden 256  --lr 0.01 --dropout 0.6 --alpha 0 --beta 100 --gamma 1.0 --tau 0.8 --lam 0.1
python test-flan.py --dataset UF21 --epochs 2000 --hidden 256  --lr 0.01 --dropout 0.6 --alpha 0 --beta 100 --gamma 0.1 --tau 0.8 --lam 0.1
python test-flan.py --dataset Hamilton46 --epochs 2000 --hidden 256  --lr 0.01 --dropout 0.6 --alpha 0 --beta 1000 --gamma 0.1 --tau 0.8 --lam 0.2
python test-flan.py --dataset Tulane29 --epochs 2000 --hidden 256  --lr 0.01 --dropout 0.6 --alpha 0 --beta 100 --gamma 0.1 --tau 0.8 --lam 0.1



