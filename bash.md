nohup python3 server.py --config_path './config_files/' --config config_covid.json &

nohup python3 client_chexpert.py --config_path './config_files/' --config 'client_chexpert.json' > client_chexpert.txt & 
nohup python3 client_chestxray.py --config_path './config_files/' --config 'config_chestxray.json' > client_chestxray.txt &

nohup python3 paper_retrieval_training_phase1.py.py --config_path './config_files/' --config 'config_chestxray.json' > chestxray_p1.txt &

nohup python3 combine_phase1.py --config_path './config_files/' --config 'config_combine.json' > combine_p1.txt &

nohup python3 server.py --config_path './config_files/' --config 'config_covid.json' > server_log.txt &

server - 3 xpert - 5 xray - 6 combine - 7

nohup python3 chestxray_phase2.py --config_path ./config_files/ --config config_chestxray.json > chestxray_phase2.txt &

nohup python3 chexpert_phase2.py --config_path ./config_files/ --config client_chexpert.json > chexpert_phase2.txt &

FL -> ckps/server_checkpoint_final combined -> models/combined_phase2

nohup python3 chestxray_phase1.py --config_path './config_files/' --config 'config_chestxray.json' > chestxray_phase1.txt &

nohup python3 chestxpert_phase1.py --config_path './config_files/' --config 'client_chexpert.json' > chestxray_phase1.txt &