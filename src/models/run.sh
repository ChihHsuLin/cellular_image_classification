#source activate cellular_classification
python train.py -y config/train_ef3_v2.yml -log
python fine_tune.py -y config/train_ef3_ft_v2.yml -log
python inference_tta.py -y config/test_ft.yml -log
python ensemble_exp.py