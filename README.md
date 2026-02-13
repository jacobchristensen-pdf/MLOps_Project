# MLOps_Project
Projekt til MLOps

Stå i mappe MLOps_Project skriv "python -m src.train"

DATA:

hundebillede i train 11702 = ødelagt
kattebillede i train 666 = ødelagt

2 databasefiler som ikke er nødvendige er også fjernet
Thumbs.db fil.

Split sket med seed 42 og med program split_data.py

Split er som følger 70/20/10 | train/test/val



For at sørge for envs er de samme kør
conda env create -f environment_full.yml

Alternativt er der en pip freeze
