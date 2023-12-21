import os
import re
import shutil
import gensim
import pandas as pd
import tensorflow as tf
from collections import Counter
import mysql.connector

# Define the model as a global variable
model = None
model_cnn = None

# Information dictionaries and lists
categories_info = {"duplicates": [], "not_in_model": []}
renaming_info = []
not_found_directories = []

# Define your MySQL database configuration
mysql_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '1234',
    'database': 'myapp'
}

def load_cnn_model():
    global model_cnn
    
    model_path = os.path.join(os.path.dirname(__file__), 'model_cnn_v1')
    model_cnn = tf.keras.models.load_model(model_path)
    
def load_word2vec_model():
    global model

    # Load the model from the correct path
    model_path = os.path.join(os.path.dirname(__file__), 'GoogleNews-vectors-negative300.bin')
    model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)

def insert_renaming_info(cursor, original_category, modified_category, status):
    # Check if the directory names already exist in the table
    cursor.execute("""
        SELECT original_category, modified_category
        FROM renaming_info
        WHERE original_category = %s AND modified_category = %s
    """, (original_category, modified_category))

    existing_entry = cursor.fetchone()

    if existing_entry is None:
        # Insert the renaming information into the MySQL database
        cursor.execute("""
            INSERT INTO renaming_info (original_category, modified_category, status)
            VALUES (%s, %s, %s)
        """, (original_category, modified_category, status))

def on_server_startup():
    global model

    if model is None:
        # Load the model if it hasn't been loaded yet
        load_word2vec_model()
        
    if model_cnn is None:
        # Load the model if it hasn't been loaded yet
        load_cnn_model()
    
    # Define the directory path
    folder_path = os.path.join(os.path.dirname(__file__), '..', 'static', 'svg')

    # Initialize the MySQL connection
    conn = mysql.connector.connect(**mysql_config)
    cursor = conn.cursor()

    renaming_info = {
        "Renamed": [],
        "No Change": [],
        "Not Found": [],
    }

    # Some directories won't be compatible with the model unless their names are adjusted, so adjustments are made here
    folder_mapping = {
        # small change was needed
        'santa claus': 'santa',

        # spaces/- get removed because then the model will find the words when these are removed
        'paper clip': 'paperclip',
        'sponge bob': 'spongebob',
        'race car': 'racecar',
        'hot-dog': 'hotdog',

        # similar directories get merged because they are too similar for the model to understand
        'floor lamp': 'lamp',
        'tablelamp': "lamp",
        'flying bird': 'bird',
        'standing bird': 'bird',
        'pickup truck': 'truck',

        # synonyms where needed to be able to run the model on these categories
        'axe': 'hatchet',
        'beer-mug': 'tankard',
        'computer-mouse': 'clicker',
        'frying-pan': 'skillet',
        'head-phones': 'earphones',
        'human-skeleton': 'cadaverous',
        'speed-boat': 'motorboat',
        't-shirt': 'tee',
        'teddy-bear': 'plushie',
        'tennis-racket': 'racket',
        'wine-bottle': 'wine',
        'wrist-watch': 'watch',
        'flying saucer': 'spaceship',
        'hot air balloon': 'aerostat',
        'palm tree': 'palmetto',
        'parking meter': 'kiosk',
        'bottle opener': 'corkscrew',
        'satellite dish': 'parabolic'
    }

    for original_folder, modified_folder in folder_mapping.items():
        # Check if the entry already exists in the database
        cursor.execute("SELECT COUNT(*) FROM renaming_info WHERE original_category = %s", (original_folder,))
        count = cursor.fetchone()[0]

        if count == 0:
            # Insert the entry if it doesn't exist
            cursor.execute("""
                INSERT INTO renaming_info (original_category, modified_category, status)
                VALUES (%s, %s, %s)
            """, (original_folder, modified_folder, "Renamed"))
            conn.commit()

    # Create the new folders if they don't exist
    new_folders = set(folder_mapping.values())
    for new_folder in new_folders:
        new_folder_path = os.path.join(folder_path, new_folder)
        os.makedirs(new_folder_path, exist_ok=True)

    # Move the content of the old folders to the corresponding new folders
    for old_folder, new_folder in folder_mapping.items():
        old_folder_path = os.path.join(folder_path, old_folder)
        modified_category = folder_mapping.get(old_folder)
        new_folder_path = os.path.join(folder_path, modified_category)

        if os.path.exists(old_folder_path):
            for item in os.listdir(old_folder_path):
                source_item = os.path.join(old_folder_path, item)
                if os.path.isfile(source_item):
                    target_item = os.path.join(new_folder_path, item)
                    shutil.move(source_item, target_item)

        # Remove the old folders
        if os.path.exists(old_folder_path):
            shutil.rmtree(old_folder_path, ignore_errors=True)

    # Get all the categories in the 'svg' folder
    categories_1 = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

    # Making sure the categories can be used for the model
    def process_category(category):
        category = category.lower()
        if "-" in category:
            category_parts = category.split('-')
            if len(category_parts) > 1:
                return category_parts[-1].strip()

        category = re.sub(r'\([^)]*\)', '', category)
        words = category.split()
        if len(words) == 2:
            return " ".join(words[1:])
        elif len(words) > 2:
            return words[0]

        return category.strip()

    categories_2 = [process_category(category) for category in categories_1]

    # Check if the categories are in the model or not
    if model:
        categories = [category for category in categories_2 if category in model]
        categories_not_in_model = [category for category in categories_2 if category not in model]
    else:
        categories = []
        categories_not_in_model = categories_2

    word_counts = Counter(categories)
    duplicates = [word for word, count in word_counts.items() if count > 1]

    categories_info["duplicates"] = duplicates
    categories_info["not_in_model"] = categories_not_in_model

    modified_folder_names = list(folder_mapping.values())

    # Loop through the original and modified category names
    for original_category, modified_category in zip(categories_1, categories):
        original_directory_path = os.path.join(folder_path, original_category)
        modified_directory_path = os.path.join(folder_path, modified_category)

        if os.path.exists(original_directory_path):
            if original_category != modified_category:
                os.rename(original_directory_path, modified_directory_path)
                status = "Renamed"
            else:
                status = "No Change"
        else:
            status = "Not Found"
            not_found_directories.append(original_category)

        if original_category not in modified_folder_names:
            insert_renaming_info(cursor, original_category, modified_category, status)

    conn.commit()
    conn.close()

    if not_found_directories:
        not_found_df = pd.DataFrame(not_found_directories, columns=["Not Found Directories"])
    else:
        not_found_df = pd.DataFrame(columns=["Not Found Directories"])

    return categories_info, renaming_info, not_found_df