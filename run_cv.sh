cd CV

"""
--use_highway To use a model with branch layers
--pretrained Use a pre-trained ResNet
--train_highway_only To use only the memory to train the branch classifiers, alternative is to use only data from the current task
--mem_per_tasks Amount of element per task to save in memory
--only_mem To use only the memory to train the model
"""

python train.py --use_highway --pretrained --train_highway_only --mem_per_tasks 100 --only_mem
