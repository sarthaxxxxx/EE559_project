$ echo Original dataset w/o no conditioning of encoded dates as a preliminary experiment
python3 main.py --train_mode whole --model base
python3 main.py --train_mode whole --model trivial
python3 main.py --train_mode whole --model knn --k 10
python3 main.py --train_mode whole --model naive 
python3 main.py --train_mode whole --model dt
python3 main.py --train_mode whole --model rf --rf_max_depth 15 --n_estimators 300
python3 main.py --train_mode whole --model svm
python3 main.py --train_mode whole --model perceptron

$ echo Original dataset by conditioning
python3 main.py --train_mode whole --model base --if_time
python3 main.py --train_mode whole --model trivial --if_time
python3 main.py --train_mode whole --model knn --k 10 --if_time
python3 main.py --train_mode whole --model naive --if_time
python3 main.py --train_mode whole --model dt --if_time
python3 main.py --train_mode whole --model rf --if_time --rf_max_depth 15 --n_estimators 300
python3 main.py --train_mode whole --model svm --if_time
python3 main.py --train_mode whole --model perceptron --if_time

$ echo Feature engineered w/o conditioning
python3 main.py --train_mode stats --model base
python3 main.py --train_mode stats --model trivial
python3 main.py --train_mode stats --model knn --k 4
python3 main.py --train_mode stats --model naive
python3 main.py --train_mode stats --model dt
python3 main.py --train_mode stats --model rf --rf_max_depth 5 --n_estimators 100
python3 main.py --train_mode stats --model svm
python3 main.py --train_mode stats --model perceptron 

$ echo Feature engineered w/ conditioning
python3 main.py --train_mode stats --model base --if_time
python3 main.py --train_mode stats --model trivial --if_time
python3 main.py --train_mode stats --model knn --k 4 --if_time
python3 main.py --train_mode stats --model naive --if_time
python3 main.py --train_mode stats --model dt --if_time
python3 main.py --train_mode stats --model rf --if_time --rf_max_depth 5 --n_estimators 100
python3 main.py --train_mode stats --model svm --if_time
python3 main.py --train_mode stats --model perceptron --if_time