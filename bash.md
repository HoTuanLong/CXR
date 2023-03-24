nohup python3 paper_retrieval_test_covid_data.py --config_path './config_files/' --config config_covid.json &

nohup python3 client_chexpert.py --config_path './config_files/' --config 'client_chexpert.json' > client_chexpert.txt &
nohup python3 client_chestxray.py --config_path './config_files/' --config 'config_chestxray.json' > client_chestxray.txt &

nohup python3 paper_retrieval_training_phase1.py.py --config_path './config_files/' --config 'config_chestxray.json' > chestxray_p1.txt &

nohup python3 combine_phase1.py.py --config_path './config_files/' --config 'config_combine.json' > combine_p1.txt &