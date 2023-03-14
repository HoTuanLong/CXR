#! /bin/bash

nohup python3 client_chexpert.py --config_path './config_files/' --config 'client_chexpert.json' > client_chexpert.txt &
nohup python3 client_chestxray.py --config_path './config_files/' --config 'client_chestxray.json' > client_chestxray.txt &
nohup python3 server.py --config_path './config_files/' --config 'config.json' > server_log.txt &