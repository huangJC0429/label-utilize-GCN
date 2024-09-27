
---for flan---- # --tau 0.8 --lam 0.1
python test-flan.py --dataset cora --epochs 2000 --hidden 8  --lr 0.015 --dropout 0.6 --alpha 0 --beta 1000 --gamma 0.1 --tau 0.2 --lam 1.0

python test-flan.py --dataset citeseer --epochs 2000 --hidden 32  --lr 0.01 --dropout 0.6 --alpha 0 --beta 1000 --gamma 0.4 --tau 0.2 --lam 0.1

python test-flan.py --dataset pubmed --epochs 2000 --hidden 4  --lr 0.02 --dropout 0.6 --alpha 0 --beta 1000 --gamma 0.5 --tau 0.8 --lam 0.1 # use GCN aligned


python test-flan-coauthor.py --dataset Computers --epochs 2000 --dropout 0.6 --lr 0.01 --hidden 256 --alpha 0 --beta 1000 --gamma 0.3 --tau 0.8 --lam 0.3 --edge_rate2 0.9

python test-flan-coauthor.py --dataset Photo --epochs 2000 --dropout 0.6 --lr 0.01 --hidden 256 --alpha 0 --beta 1000 --gamma 0.1 --tau 0.8 --lam 0.6 --edge_rate2 0.9


-----heherophily----

python test-flan-hetero.py --dataset chameleon --epochs 2000 --dropout 0.6 --lr 0.005 --hidden 256 --alpha 0 --beta 1000 --gamma 0.3 --tau 0.8 --lam 0.1 --edge_rate2 0.9 --weight_decay 0.0

python test-flan-hetero.py --dataset squirrel --epochs 2000 --dropout 0.5 --lr 0.02 --hidden 256 --alpha 0 --beta 1000 --gamma 0.1 --tau 0.8 --lam 0.6 --edge_rate2 0.9 --weight_decay 0.0