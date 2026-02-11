import os
from collections import defaultdict
from sklearn.model_selection import train_test_split

def normalize_label(label):
    """Normalize labels to lowercase and stripped"""
    return label.strip().lower()

def load_file_groups(dir_v0, dir_v12, label_to_int):
    """Load and group files by version"""
    filenames_v0 = os.listdir(dir_v0)
    filenames_v12 = os.listdir(dir_v12)
    
    file_groups = {}
    
    # Process v0 files
    for file in filenames_v0:
        if "@" in file:
            label = normalize_label((file.split("@")[1]).split('.')[0])
            if label in label_to_int:
                base_name = file.split("@")[0]
                if base_name not in file_groups:
                    file_groups[base_name] = {"v0": None, "v1": None, "v2": None}
                file_groups[base_name]["v0"] = file
    
    # Process v12 files
    for file in filenames_v12:
        if "@" in file:
            label = normalize_label((file.split("@")[1]).split('.')[0])
            if label in label_to_int:
                base_name = file.split("@")[0].split("_", 1)[1]
                if base_name not in file_groups:
                    file_groups[base_name] = {"v0": None, "v1": None, "v2": None}
                if file.startswith("v1_"):
                    file_groups[base_name]["v1"] = file
                elif file.startswith("v2_"):
                    file_groups[base_name]["v2"] = file
    
    return file_groups

def read_files_and_append(file_group, code_list, filename_list, base_path_v0, base_path_v12):
    """Read files from disk and append to lists"""
    for version in ['v0', 'v1', 'v2']:
        file = file_group[version]
        if file:
            if version == 'v0':
                with open(os.path.join(base_path_v0, file), 'r') as buggy_file:
                    code_list.append(buggy_file.read())
            else:
                with open(os.path.join(base_path_v12, file), 'r') as buggy_file:
                    code_list.append(buggy_file.read())
            filename_list.append(file)

def count_categories_and_versions(filenames, label_to_int):
    """Count categories and versions in filenames"""
    category_counts = defaultdict(lambda: defaultdict(int))
    for file in filenames:
        if "@" in file:
            label = normalize_label((file.split("@")[1]).split('.')[0])
            if label in label_to_int:
                version = 'v2' if 'v2' in file else 'v1' if 'v1' in file else 'v0'
                category_counts[label][version] += 1
    return category_counts

def prepare_data(dir_v0, dir_v12, label_to_int, test_size=0.2, random_state=42):
    """Prepare train and validation datasets"""
    file_groups = load_file_groups(dir_v0, dir_v12, label_to_int)
    
    # Split groups into train and test sets
    group_keys = list(file_groups.keys())
    train_keys, test_keys = train_test_split(group_keys, test_size=test_size, 
                                             random_state=random_state)
    
    train_buggy_code = []
    valid_buggy_code = []
    train_filenames = []
    valid_filenames = []
    
    # Load training data
    for key in train_keys:
        if key not in file_groups:
            continue
        try:
            read_files_and_append(file_groups[key], train_buggy_code, 
                                train_filenames, dir_v0, dir_v12)
        except FileNotFoundError as e:
            print(f"Skipping missing file for key {key}: {e}")
            continue
    
    # Load validation data
    for key in test_keys:
        if key not in file_groups:
            continue
        try:
            read_files_and_append(file_groups[key], valid_buggy_code, 
                                valid_filenames, dir_v0, dir_v12)
        except FileNotFoundError as e:
            print(f"Skipping missing file for key {key}: {e}")
            continue
    
    # Count categories
    train_category_counts = count_categories_and_versions(train_filenames, label_to_int)
    valid_category_counts = count_categories_and_versions(valid_filenames, label_to_int)
    
    return (train_buggy_code, train_filenames, valid_buggy_code, valid_filenames,
            train_category_counts, valid_category_counts)