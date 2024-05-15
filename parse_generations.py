import os
import re
import json

folder = '/home/mashalimay/webarena/visualwebarena/results/google/gemini-pro-1.0-vision-completion'
save_folder = f"/generations/"

def parse_generation_data(txt_file):
    with open(txt_file, 'r') as f:
        lines = f.readlines()

    generation_data = []
    recording = False
    for line in lines:
        if line.startswith('GENERATION'):
            recording = True

        elif recording:
            if line.strip() == '----------------------------------':
                pass
            else:
                generation_data.append(line.strip())
                recording = False
    return generation_data

# Iterate over experiments in this group
for subfolder_name in os.listdir(folder):
    # Get subfolder path
    subfolder_path = os.path.join(folder, subfolder_name)

    # Iterate over conversation files for tasks in this experiment
    if os.path.isdir(subfolder_path):
        generation_data = {}

        # Iterate over conversation files in this task
        for conversation_file in os.listdir(subfolder_path):
            if conversation_file.endswith(".txt"):
                # Parse task_id from filename
                match = re.match(r"conversations_(\d+).txt", conversation_file)
                task_id = match.group(1) if match else None
                if task_id is None:
                    continue
                
                # Extract generation data
                generation_data[int(task_id)] = parse_generation_data(os.path.join(subfolder_path, conversation_file))

            # Save to JSON file
        with open(os.path.join(subfolder_path, f'generations.json'), 'w') as json_file:
            json.dump(generation_data, json_file, indent=4)

