import os
import re
import cv2
import random
import shutil
import cairosvg
import numpy as np
import pandas as pd
import mysql.connector
import scripts.startup_script
from werkzeug.datastructures import FileStorage
from flask import Flask, render_template, request, flash, Response
from scripts.startup_script import (categories_info, renaming_info, on_server_startup)


# setting app name and secret key
app = Flask(__name__)
app.secret_key = 'this_is_a_secret_key'

# credientials for the mysql database
mysql_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '1234',
    'database': 'myapp'
}

# Initialize variables with data from the startup script
categories_info, renaming_info, not_found_directories = on_server_startup()

# getting the word2vec model from the startup script
model = scripts.startup_script.model
model_cnn = scripts.startup_script.model_cnn

# looking at the variabels from the start up script and seeing if there is anything in it if not the variable will be set to true
no_duplicates = len(categories_info['duplicates']) == 0
all_in_model = len(categories_info['not_in_model']) == 0
all_directories_found = len(not_found_directories) == 0

# allowed file extions to be submitted
allowed_file_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tga', '.pcx', '.ppm', '.pgm', '.pbm']

# list for the directories when there added to the database
added_directories_info = []

# getting the base dict
base_directory = os.path.dirname(__file__)
svg_folder = os.path.join(base_directory,'static', 'svg')

# starting the cam
cap = cv2.VideoCapture(0)

# home route
@app.route('/')
def home():
    # Update the database with renaming info, to make sure that all the data is present
    update_database()

    # calling for the data from the database
    renaming_info_from_db = get_renamed_directories()
    added_info = get_added()
    
    # getting the legnt of the variabels and putting the numbers into a list so we can use it in a tabel later
    index_numbers = list(range(1, len(get_renamed_directories()) + 1))
    index_numbers_added = list(range(1, len(get_added()) + 1))
    
    # calculating the amount of categories
    categories = get_categories()
    len_categories = len(categories)
        
    # return all these varaibels to the html page 
    return render_template('home.html',
                           categories_info=categories_info,
                           len_categories=len_categories,
                           
                           renaming_info=renaming_info,
                           renamed_directories=renaming_info_from_db,
                           added_info=added_info,
                           
                           index_numbers=index_numbers,
                           index_numbers_added=index_numbers_added,
                           
                           not_found_directories=not_found_directories,
                           no_duplicates=no_duplicates,
                           
                           all_in_model=all_in_model,
                           all_directories_found=all_directories_found
                           
                           )


# route for the upload page
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    # get all the categories
    categories = get_categories()
    
    # looking if post is used
    if request.method == 'POST':
        # getting the informating from the form
        new_directories = request.files.getlist('folder')
        
        # making sure that all folders are save corretly
        for new_directory in new_directories:
            is_valid_folder(new_directory)
         
        # getting the base_directory from where this file is located and then fuse the path with a new folder   
        tmp_directory = os.path.join(base_directory, 'tmp')
        os.makedirs(tmp_directory, exist_ok=True)
        
        tmp_directory_2 = os.path.join(base_directory, 'tmp2')
        os.makedirs(tmp_directory_2, exist_ok=True)
        
        # get the diretory tree for every item in the tmp2 folder
        for item in os.listdir(tmp_directory_2):
            tmp_directory_2_item = os.path.join(tmp_directory_2, item)
            if os.path.isdir(tmp_directory_2_item):
                directory_tree = list_directory_tree(tmp_directory_2_item)

        # Ensure the directory structure complies with the allowed depth
        if not has_deep_directory_structure(directory_tree):
            flash("Directory structure does not comply with the allowed depth.")
            # because the folder structure isnt allowed we will clean the tmp directories 
            shutil.rmtree(tmp_directory, ignore_errors=True)
            shutil.rmtree(tmp_directory_2, ignore_errors=True)
        # if it complies continue here
        else:
            # we dont need this tmp folder anymore so we will delete it
            shutil.rmtree(tmp_directory_2, ignore_errors=True)
            # creating a list for matched folders
            matched_directories = []
            
            # were gonna clean the directory names to make sure the model can understand them, where gonna do this for every dict in tmp
            for old_dict in os.listdir(tmp_directory):
                temp_directory_with_item = os.path.join(tmp_directory, old_dict)

                # Process the category name
                cleaned_directory_name = process_category(old_dict)
                
                # Check if the cleaned_directory_name is in the categories list, if so add it to the list from earlier
                if cleaned_directory_name in categories:
                    matched_directories.append(cleaned_directory_name)
                # if the list contains anything return this template
                if matched_directories:
                    shutil.rmtree(tmp_directory, ignore_errors=True)
                    return render_template('confirm_matching_directories.html', matched_directories=matched_directories)   
                # if the list is empty continue here        
                else:
                    # check if the model is loaded in 
                    if model:
                        # check if the cleaned names are NOT in the model
                        dict_not_in_model = [category for category in cleaned_directory_name if category not in model]
                        # if the list contains anything if so flash this:
                        if dict_not_in_model:
                            flash("Category name isn't in the model change these names:")
                            # show the names of the dicts that are not in the model
                            for dicts in dict_not_in_model:
                                flash(f"- {dicts}")
                            # delete the dicts that are not in in the model
                            shutil.rmtree(temp_directory_with_item, ignore_errors=True)
                        # if the list is empty continue here
                        else:
                            # check if the files in the items from the tmp dict are allowed, if not delete them
                            delete_files_with_disallowed_extensions(temp_directory_with_item, allowed_file_extensions)
                            
                            # get the names from all dicts currently in the svg dict
                            pre_existing_directories = set(os.listdir(svg_folder))
                            
                            # create the new dict where the files will be moved too
                            dest_dir = os.path.join(svg_folder, cleaned_directory_name)
                            os.makedirs(dest_dir, exist_ok=True)
                            
                            # Process the files in the top-level directory
                            process_directory(temp_directory_with_item, dest_dir)
                            # Process subdirectories within the top-level directory
                            process_subdirectories(temp_directory_with_item, dest_dir)
                            
                            # check what dicts where added to the svg dict
                            added_directories = set(os.listdir(svg_folder)) - pre_existing_directories
                            # get the path to the added dicts
                            for directory in added_directories:
                                directory_path = os.path.join(svg_folder, directory)
                                # check if the path acctully exists if so get some info from it
                                if os.path.isdir(directory_path):
                                    num_files = len(os.listdir(directory_path))
                                    added_directories_info.append((directory, num_files))
                                    add_new_categories_to_database(old_dict, directory)
                            # check if the list contains anything if so flash that it worked and the dict name and file number
                            if added_directories_info:
                                flash("Processing successful. Added directories:")
                                for directory, num_files in added_directories_info:
                                    flash(f"- {directory}: {num_files} files")
                    else:
                        flash("model isn't loaded")
                        return render_template('upload.html')
            # make sure that we delete the tmp dict now that everthing has been moved
            shutil.rmtree(tmp_directory, ignore_errors=True)
                    
    return render_template('upload.html')


# you get here if there is a matching dict found
@app.route('/confirm_matching_directories', methods=['POST'])
def confirm_match():
    # checks the request method and get info out the form if it is post
    if request.method == 'POST':
        matched_directory = request.form['matched_directory']
        old_directory_name = request.form['old_directory_name']
        confirm = request.form.get('confirm')

        # look if the confirm from the form is yes, if yes move the files to the existing dict(categorie)
        if confirm == 'yes':
            # Implement the file moving logic
            move_files_to_existing_category(matched_directory, old_directory_name)
            flash("Files moved to the existing category successfully.")
        # if it is not yes just return the upload page so then can try to upload again if they want
        else:
            return render_template('/upload.html')


@app.route('/cam_feed')
def cam_feed():
    return render_template('video_feed.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/capture_photo', methods=['POST'])
def capture_photo():
    # Capture a single frame
    success, frame = cap.read()
    if success:
        # Save the frame to the static/test folder
        photo_path = os.path.join('static', 'test', 'captured_photo.jpg')
        cv2.imwrite(photo_path, frame)
        
        output, svg_svg, saved_image = get_output()
        return render_template('output.html', output=output, svg_svg=svg_svg, saved_image=saved_image)
    else:
        return render_template('video_feed.html')


@app.route('/output')
def output():
    output, svg_svg, saved_image = get_output()
    return render_template('output.html', output=output, svg_svg=svg_svg, saved_image=saved_image)


def get_output():
    output = []
    
    categories = get_categories()
    
    saved_image = os.path.join('test', 'captured_photo.jpg').replace(os.sep, '/')
    img_to_use = os.path.join('static', saved_image)
    image = cv2.imread(img_to_use, cv2.IMREAD_GRAYSCALE)
    
    # Adjust contrast and brightness
    alpha = 1.75  # Contrast control (1.0 means no change)
    beta = 10   # Brightness control (0 means no change)

    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    blurred = cv2.GaussianBlur(adjusted_image, (5, 5), 0)

    # Adjust the threshold values
    lower_threshold = 50
    upper_threshold = 69

    edges = cv2.Canny(blurred, lower_threshold, upper_threshold)

    # Find contours in the binary edge image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    limitInPercent = .1
    totalSizeImage = image.shape[0] * image.shape[1]

    boundingBoxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Check if bounding box is at edges
        if (x == 0 or y == 0 or x == image.shape[0] or y == image.shape[1]):
            continue
        # Check if bounding box is too small (exmaples: dust, small dots)
        if ((w * h) < totalSizeImage * (limitInPercent/100)):
            continue
        boundingBoxes.append((x,y,w,h))

        newimg = cv2.imread(img_to_use, cv2.IMREAD_GRAYSCALE)

    padding = 5

    boundingBoxes = pd.DataFrame(boundingBoxes)
    xMin = boundingBoxes.iloc[boundingBoxes[0].idxmin()][0] - padding
    xMax = boundingBoxes.iloc[boundingBoxes[0].idxmax()][0] + boundingBoxes.iloc[boundingBoxes[0].idxmax()][2] + padding
    yMin = boundingBoxes.iloc[boundingBoxes[1].idxmin()][1] - padding
    yMax = boundingBoxes.iloc[boundingBoxes[1].idxmax()][1] + boundingBoxes.iloc[boundingBoxes[1].idxmax()][3] + padding

    cropped_image = newimg[yMin:yMax, xMin:xMax]
    cv2.imwrite("static/test/cropped_image.png", cropped_image)
    cropped = "static/test/cropped_image.png"
    
    img_array = cv2.imread(cropped)
    img_array = cv2.resize(img_array, (150, 150))  # Wijzig de formaat van de foto met OpenCV
    cv2.imwrite("static/test/img_image.png", img_array)
    
    img = img_array.flatten()
    flat_data = np.array(img)
    print(flat_data)
        
    result = model_cnn.predict(flat_data.reshape(-1, 150, 150, 3))
    result_max = np.argmax(result)
        
    categorie = categories[result_max]
    output.append(f"Predicted {categorie}")
    
    # Simulate the output of the SVM by picking a random category and removing it from the list
    random_cat = result_max
    pre = categories.pop(random_cat)
 
    # Find closest words to the randomly selected category
    similarities = {word: model.similarity(pre, word) for word in categories}
    sorted_words = sorted(similarities, key=similarities.get, reverse=True)
    closest_words = sorted_words[:3]

    # Randomly select one word from the closest words
    selected_word = random.choice(closest_words)

    # Add this to the output
    output.append(f"Closest word to {pre} is: {selected_word}")

    # Find the directory path of the selected word
    directory_path = os.path.join(svg_folder, selected_word)
    # Check if there are files in the directory path
    files_in_directory = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

    # Grab a random file from the files in the directory
    if files_in_directory:
        random_file = random.choice(files_in_directory)
        svg_svg = os.path.join('svg', selected_word, random_file).replace(os.sep, '/')
        svg_image = os.path.join(directory_path, random_file)
        output.append(f"Selected file for {selected_word}: {random_file}")
        output.append(f"Relative path from the excute file to selected file: (static/{svg_svg})")
        output.append(f"Absolute path to selected file: ({svg_image})")

    categories.append(pre)

    return output, svg_svg, saved_image


def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# make sure that the database is updated with the info from the start up script
def update_database():
    conn = mysql.connector.connect(**mysql_config)
    cursor = conn.cursor()
    # get the info from the dict
    for info in renaming_info["Renamed"]:
        original_category = info["Original Category"]
        modified_category = info["Modified Category"]
        status = info["Status"]
        
        # add the info to the database
        cursor.execute("""
            INSERT INTO renaming_info (original_category, modified_category, status)
            VALUES (%s, %s, %s)
        """, (original_category, modified_category, status))

    conn.commit()
    conn.close()


def get_categories():
    categories_1 = [f for f in os.listdir(svg_folder) if os.path.isdir(os.path.join(svg_folder, f))]
    return categories_1


# getting the info from the database where the status is renamed
def get_renamed_directories():
    conn = mysql.connector.connect(**mysql_config)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT original_category, modified_category
        FROM renaming_info
        WHERE status = 'Renamed'
    """)
    renamed_directories = cursor.fetchall()
    conn.close()
    return renamed_directories


# add to the database the categories that have just been added with status added
def add_new_categories_to_database(original_category, modified_category):
    conn = mysql.connector.connect(**mysql_config)
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO renaming_info (original_category, modified_category, status) VALUES (%s, %s, %s)", 
        (original_category, modified_category, "Added")
        )

    conn.commit()
    conn.close()


# get the info from the database where the stats is added
def get_added():
    conn = mysql.connector.connect(**mysql_config)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT original_category, modified_category
        FROM renaming_info
        WHERE status = 'added'
    """)
    added_info = cursor.fetchall()
    conn.close()
    return added_info


# just add the content from the folder to these tmp/tmp2 folders, 
# the files in tmp2 will containe no info that is normale we dont need it to containe anything just want the dict structure and file names (without we messing with the structure)
def is_valid_folder(content):
    temp_directory = os.path.join(base_directory, 'tmp')
    os.makedirs(temp_directory, exist_ok=True)
    temp_directory_2 = os.path.join(base_directory, 'tmp2')
    os.makedirs(temp_directory_2, exist_ok=True)

    #check if the content is in the filestorage format
    if isinstance(content, FileStorage):
        filename = content.filename
        filename_parts = filename.split('/')
        
        # Check if there are more than 2 elements
        if len(filename_parts) > 2:
            # Remove the first element
            filename_parts.pop(0)
            # Join the remaining parts back together with slashes
            modified_filename = '/'.join(filename_parts)
        else:
            # If there's only 2 parts, keep the original filename
            modified_filename = filename
        
        # save the content to the tmp dict
        uploaded_file_path = os.path.join(temp_directory, modified_filename)
        os.makedirs(os.path.dirname(uploaded_file_path), exist_ok=True)
        content.save(uploaded_file_path)
        
        # save the content to the tmp2 dict, these files will be empty but that is fine we only need them for there file paths
        uploaded_file_path = os.path.join(temp_directory_2, content.filename)
        os.makedirs(os.path.dirname(uploaded_file_path), exist_ok=True)
        content.save(uploaded_file_path)
        
        return True

    return False


# get the diretory path so we can rum some checks on it to make sure we can use it
def list_directory_tree(directory_path):
    tree = []
    for root, directories, files in os.walk(directory_path):
        # Ensure we're not going deeper than two levels
        if root.count(os.path.sep) - directory_path.count(os.path.sep) >= 2:
            continue
        
        directory_info = {
            "root": root,
            "directories": directories,
            "files": files
        }
        tree.append(directory_info)
    return tree


# use the info we gatherd with the list_directory_tree function and run some checks on it
def has_deep_directory_structure(tree):
    for directory_info in tree:
        contains_subdirectories = bool(directory_info["directories"])
        contains_files = bool(directory_info["files"])

        # Check if both files and subdirectories exist in the same root directory
        if contains_subdirectories and contains_files:
            return False

        # Check if there are subdirectories within subdirectories
        if contains_subdirectories:
            for sub_directory in directory_info["directories"]:
                if os.path.join(directory_info["root"], sub_directory).count(os.path.sep) - directory_info["root"].count(os.path.sep) > 1:
                    return False

    return True


# check the file extension if not allwed just remove the file
def delete_files_with_disallowed_extensions(directory_path, allowed_extensions):
    for root, directories, files in os.walk(directory_path):
        for file in files:
            file_extension = os.path.splitext(file)[1].lower()
            file_path = os.path.join(root, file)
            
            if file_extension not in allowed_extensions and file_extension != ".svg":
                os.remove(file_path)


# the file extension was not svg so we need to conver it we will try to do that here     
def convert_to_svg(input_file, output_file):
    try:
        cairosvg.svg2svg(url=input_file, write_to=output_file)
        # After successful conversion, delete the original image file
        os.remove(input_file)
    except Exception as e:
        print(f"Error converting {input_file} to SVG: {str(e)}")


# here where processes the directorys directly under the tmp folder       
def process_directory(directory_path, dest_dir):
    for root, directories, files in os.walk(directory_path):
        for file in files:
            file_extension = os.path.splitext(file)[1].lower()
            input_file = os.path.join(root, file)
            
            if file_extension == ".svg":
                # If the file is already an SVG, move it to the destination directory.
                output_file = os.path.join(dest_dir, file)
                shutil.move(input_file, output_file)
            elif file_extension in allowed_file_extensions:
                # If it's not an SVG and has an allowed extension, convert it to SVG.
                output_file = os.path.join(dest_dir, f"{os.path.splitext(file)[0]}.svg")
                convert_to_svg(input_file, output_file)

                
# processes the subdicts, after the latest update to tis file there should be no subdicts after the tree checkes but just to be sure, i'll leave it here for now
def process_subdirectories(base_directory, dest_dir):
    for root, directories, files in os.walk(base_directory):
        for directory in directories:
            dir_path = os.path.join(root, directory)
            process_directory(dir_path, dest_dir)
            shutil.rmtree(dir_path)


# process to clean the category names (is also used in the startup script)      
def process_category(category):
    # make sure everthin is lower case
    category = category.lower()
    # check if there are - in the strings
    if "-" in category:
        # if so split the string
        category_parts = category.split('-')
        # if there are more then 1 part after the strip we strip the first part away
        if len(category_parts) > 1:
            return category_parts[-1].strip()
        
    # remove () and its content
    category = re.sub(r'\([^)]*\)', '', category)
    
    # look if there are mutiple words in the string, if there are 2 then delete the first
    words = category.split()
    
    if len(words) == 2:
        return " ".join(words[1:])
    # if there are more then 2 just return the first
    elif len(words) > 2:
        return words[0]

    return category.strip()


# if you want to add more files to an existing categorie
def move_files_to_existing_category(existing_category, old_directory_name):
    # get all the paths
    dest_dir = svg_folder
    temp_directory = os.path.join(base_directory, 'tmp')
    source_directory = os.path.join(temp_directory, old_directory_name)
    destination_directory = os.path.join(dest_dir, existing_category)

    # Delete files with disallowed extensions
    delete_files_with_disallowed_extensions(source_directory, allowed_file_extensions)

    # Move files from source_directory to destination_directory while handling extensions
    for filename in os.listdir(source_directory):
        source_file = os.path.join(source_directory, filename)

        # Convert the source file to SVG before moving it
        destination_file = os.path.join(destination_directory, f"{os.path.splitext(filename)[0]}.svg")
        convert_to_svg(source_file, destination_file)

    # remove the source dict when done
    os.rmdir(source_directory)

if __name__ == '__main__':
    app.run()