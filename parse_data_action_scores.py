

import os
import re
import pandas as pd

folder='/home/mashalimay/webarena/visualwebarena/results/google/gemini-pro-1.0-completion'
exclude_folders = {'wordclouds', 'generations'}

def count_num_actions(txt_file):
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    
    num_actions = 0
    for line in lines:
        if line.startswith('GENERATION\n'):
            num_actions += 1
    
    return num_actions

def count_incorrect(txt_file, match_text='However, the format was incorrect.'):
    with open(txt_file, 'r') as f:
        content = f.read()
        count = content.count(match_text)
    return count

def get_scores(txt_file):
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    
    scores = {}
    prev_task_id = -1
    task_id = -1
    for line in lines:
        # Get task_id
        match = re.search(r"/(\d+)\.json", line)
        if match:
            task_id = int(match.group(1))

        if task_id > prev_task_id:
            match = re.search(r"\[Result\] \((\w+)\)", line)
            if match:
                score = 1 if match.group(1) == 'PASS' else 0
                scores[task_id] = score
                prev_task_id = task_id
        if prev_task_id == 25:
            break
    return scores

num_actions = {}
scores = {}
incorrects = {}
for f in os.listdir(folder):
    # get subfolder name
    subfolder = os.path.join(folder, f)

    if f in exclude_folders: continue

    # check if it's a directory
    if os.path.isdir(subfolder):
        num_actions[f] = {}
        incorrects[f] = {}
        scores[f] = get_scores(f"{subfolder}/log.txt")
        
        # list all txt files in the subfolder
        for conversation in os.listdir(subfolder):
            if conversation.endswith(".txt"):
                # parse task_id
                match = re.match(r"conversations_(\d+).txt", conversation)
                task_id = match.group(1) if match else None
                if task_id is None: continue
                num_actions[f][int(task_id)] = count_num_actions(os.path.join(subfolder, conversation))
                incorrects[f][int(task_id)] = count_incorrect(os.path.join(subfolder, conversation))


df_num_actions = pd.DataFrame(num_actions)
df_scores = pd.DataFrame(scores)
df_incorrects = pd.DataFrame(incorrects)

df_num_actions = df_num_actions.sort_index()
df_scores = df_scores.sort_index()
df_incorrects = df_incorrects.sort_index()

df_num_actions.to_csv(os.path.join(folder, 'num_actions.csv'))
df_scores.to_csv(os.path.join(folder, 'scores.csv'))
df_incorrects.to_csv(os.path.join(folder, 'incorrects.csv'))

