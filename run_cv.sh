cd CV

"""
--use_highway To use a model with branch layers
--mem_per_tasks Amount of element per task to save in memory
--only_mem To use only the memory to train the model
"""

python train.py --use_highway --mem_per_tasks 100 --only_mem