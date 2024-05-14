

import json
import os


path="/home/mashalimay/webarena/visualwebarena/config_files/test_webarena"

def sort_key(file_name):
    return int(file_name.split('.')[0])


# list files in the directory
files = sorted(os.listdir(path), key=sort_key)

i = 0
for f in (files):
    if f.endswith(".json"):
        config = json.load(open(os.path.join(path, f)))
        eval_ty = config['eval']["eval_types"]
        reference_answers = config['eval']["reference_answers"]
    
        # if reference_answers is not None:    
        #     ref_answers = [item for item in reference_answers.values()]
        #     flag = any(["N/A" in ans for ans in ref_answers])
        #     if flag:
        #         print("Eval type: ", eval_ty)
        #         print("Reference answers: ", reference_answers)
        #         print("File: ", f)
        #         print("\n")

        if 'string_match' in eval_ty:
            print("Eval type: ", eval_ty)
            print("Reference answers: ", reference_answers)
            print("File: ", f)
            print("\n")

