
import os
import json
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


group_ranges = {
    'shop_admin_279':[0,6],
    'map_79':[7,10],
    'shop_admin_288':[11,15],
    'map_79':[16,20],
    'shopping_222':[21,25],    
}

exclude_words = set(STOPWORDS).union(
    {"step-by-step", "step", "next", "action", "perform", "summary", "let's", "think",})


folder = '/home/mashalimay/webarena/visualwebarena/results/google/gemini-pro-1.0-vision-completion'
wordclouds_folder = f'{folder}/wordclouds'


# Ensure the save folder exists
os.makedirs(wordclouds_folder, exist_ok=True)

def load_generation_data(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def generate_word_cloud(text, save_path):
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=exclude_words).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(save_path, format='png')
    plt.close()

# Word clouds per experiment-taskgroup
for subfolder_name in os.listdir(folder):
    subfolder_path = os.path.join(folder, subfolder_name)
    json_file_path = os.path.join(subfolder_path, 'generations.json')

    generations_per_group = {}
    if os.path.isfile(json_file_path):
        generation_data = load_generation_data(json_file_path)

        # Combine task generation for each group
        for task, generations in generation_data.items():
            for group_name, (start_idx, end_idx) in group_ranges.items():
                if start_idx <= int(task) <= end_idx:
                    if group_name not in generations_per_group:
                        generations_per_group[group_name] = []
                    generations_per_group[group_name].extend(generations)

        # Generate and save word clouds
        text_per_group = {group: " ".join(generations) for group, generations in generations_per_group.items()}
        for group, text in text_per_group.items():
            save_path = os.path.join(wordclouds_folder, f'{subfolder_name}/{group}.png')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            generate_word_cloud(text, save_path)
            



#================================================================================================
# WordCloud per task
#================================================================================================
all_task_texts = {}

# Iterate over each experiment subfolder
for subfolder_name in os.listdir(folder):
    subfolder_path = os.path.join(folder, subfolder_name)
    json_file_path = os.path.join(subfolder_path, 'generations.json')

    if os.path.isfile(json_file_path):
        generation_data = load_generation_data(json_file_path)

        # Combine text for each task ID in this subfolder
        for task_id, texts in generation_data.items():
            if task_id not in all_task_texts:
                all_task_texts[task_id] = []
            all_task_texts[task_id].extend(texts)

# Generate and save word cloud for each task ID
for task_id, texts in all_task_texts.items():
    all_text = " ".join(texts)
    save_path = os.path.join(wordclouds_folder, f'{task_id}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    generate_word_cloud(all_text, save_path)


#================================================================================================
# WordCloud per group
#================================================================================================
# Aggregate text by group
group_texts = {group: [] for group in group_ranges.keys()}
for task_id, texts in all_task_texts.items():
    for group, (start_idx, end_idx) in group_ranges.items():
        if start_idx <= int(task_id) <= end_idx:
            group_texts[group].extend(texts)

# Generate and save word cloud for each group
for group, texts in group_texts.items():
    all_text = " ".join(texts)
    save_path = os.path.join(wordclouds_folder, f'{group}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    generate_word_cloud(all_text, save_path)