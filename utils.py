npy_data = "./static/embeddings/vn2.npz"

npy_class_data = "./static/embeddings/processed_embedding.npz"
import random
import string
import csv
import os
import faiss
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt
from skimage.io import imread, imsave
from time import monotonic

def add_new_person(imgs,new_id,model_interf,face_crop,name):
    """
    return a new person embedding added to the current embedding
    add new id and filenames to the system
    imgs is the list of numpy array images
    """
    # load latest data base
    # load some initial value of embeddings , filename and classes
    aa = np.load(npy_data,allow_pickle=True)
    img_names = []
    embs, imm_classes, filenames = aa["embs"], aa["imm_classes"], aa["filenames"]
    print(filenames.shape)
    embss, img_class = embs.astype("float32"), imm_classes.astype("int")
    embss_root = embss.copy()
    embss = np.load(npy_class_data)['embs']
    personal_embeddings = np.array([]).reshape(0, 512)
    for img in imgs:
        # crop then read the cropped face
        filename =  ''.join(random.choices(string.ascii_lowercase +
                             string.digits, k=8)) + '.jpg'
        img_names.append(filename)
        img_path = os.path.join("./static/images", filename)
        imsave(img_path,img)
        cropped_path = os.path.join("./static/cropped_image", filename)
        imsave(cropped_path, crop_image(img_path,face_crop))
        print(cropped_path)
        img = cv2.imread(cropped_path)
        img = cv2.resize(img, (112,112), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)
        # single embeeding
        semb = model_interf(img)
        semb = normalize(np.array(semb).astype("float32"))[0]
        personal_embeddings = np.vstack([personal_embeddings, semb])
    p_pos_embs = personal_embeddings
    p_register_base_emb = normalize([np.sum(p_pos_embs, 0)])[0]
    normalized_embeddings_new = np.vstack([embss.copy(), p_register_base_emb])
    # register successful , load new faiss with new embedding
    index = faiss_init(normalized_embeddings_new)
    #save new index
    faiss.write_index(index,"vector.index")
    # add the new id and information
    imm_classes_new = np.append(imm_classes, new_id)
    imm_classes_new = np.append(imm_classes_new, new_id)
    imm_classes_new = np.append(imm_classes_new, new_id)

    print(filenames.shape)
    # filenames = filenames.tolist()
    filename_new = np.append(filenames,os.path.basename(img_names[0]))
    filename_new = np.append(filename_new,os.path.basename(img_names[1]))
    filename_new = np.append(filename_new,os.path.basename(img_names[2]))
    filename_new = np.array(filename_new)
    print(normalized_embeddings_new.shape)
    print(filename_new.shape)
    print(imm_classes_new.shape)
    # write to csv data
    file= open('static/data.csv', 'a+', newline='')
    writer = csv.writer(file)
    writer.writerow([new_id, name, img_names[0]])
    writer.writerow([new_id, name, img_names[1]])
    writer.writerow([new_id, name, img_names[2]])
    # save new info
    np.savez(npy_data, embs=embss_root, imm_classes=imm_classes_new, filenames=filename_new)
    np.savez(npy_class_data, embs=normalized_embeddings_new)
    return normalized_embeddings_new,imm_classes_new,filename_new


def find_string_by_person_id(person_id,csv_file="./static/data.csv"):
    df = pd.read_csv(csv_file,header=None,names=['person_id','name','path'])
    person_id = person_id[0]
    filtered_df = df[df['person_id'] == person_id]
    
    if not filtered_df.empty:
        result = filtered_df.iloc[0]['name']  # Replace 'string_column' with the actual column name containing the string
        return str(result)  # Convert the result to a string
    
    return "No matching person ID found"

def faiss_init(embs,metric = 'cosine'):
    dimensions = 512 #FaceNet output is 128 dimensional vector    
    if metric == 'euclidean':
        index = faiss.IndexFlatL2(dimensions)
    elif metric == 'cosine':
        index = faiss.IndexFlatIP(dimensions)
    # add mean embeddings of previous 1000 person
    index.add(embs.astype(np.float32))
    return index

def return_id_imgs(img,model_interf,face_crop):
    # check attendance using 1 image
    # read in current data first

    aa = np.load(npy_data,allow_pickle=True)
    embs, imm_classes, filename_new = aa["embs"], aa["imm_classes"], aa["filenames"]
    # print(filename_new)
    embss, imm_classes_new = embs.astype("float32"), imm_classes.astype("int")
    embss = np.load(npy_class_data)['embs']
    filename =  ''.join(random.choices(string.ascii_lowercase +
                             string.digits, k=8)) + '.jpg'
    img_path = os.path.join("./static/images", filename)
    imsave(img_path,img)
    # embedding first
    semb = single_embedding(img_path,model_interf,face_crop) #single embedding
    k = 1
    # every time prediction, read the latest faiss index
    index = faiss.read_index("./vector.index")
    target_representation = np.array(semb, dtype='f')
    target_representation = np.expand_dims(target_representation, axis=0)
    distances, neighbors = index.search(target_representation, k)
    print(neighbors)
    print(imm_classes_new)
    id_list = np.argwhere(imm_classes_new == neighbors.tolist()[0])
    img_list = []
    for i in id_list:
        img_list.append(filename_new[i][0])
    return id_list,img_list,distances.tolist(),neighbors.tolist()[0]

def crop_image(img_path,face_crop):
    _,_,_,img = face_crop.detect_in_image(img_path)
    
    return img[0] # return the first image only

    
def single_embedding(img_path,model_interf,face_crop):
    
    filename = os.path.basename(img_path)
    cropped_path = os.path.join("./static/cropped_image", filename)
    imsave(cropped_path, crop_image(img_path,face_crop))
    img = cv2.imread(cropped_path)
    img = cv2.resize(img, (112,112), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    # single embeeding
    semb = model_interf(img)
    semb = normalize(np.array(semb).astype("float32"))[0]
    return semb

def resize_image(img, max_size=1200):


    # Get the height and width of the image
    height, width = img.shape[:2]

    # Find the longest side of the image
    longest_side = max(height, width)

    # Calculate the scale factor to resize the image
    scale_factor = max_size / float(longest_side)

    # Calculate the new dimensions
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Resize the image using the calculated dimensions
    resized_img = cv2.resize(img, (new_width, new_height))

    return resized_img

def get_max_user_id():
    # get current max_user_id
    max_id = 999  # Default value if data.csv doesn't exist or is empty
    if os.path.isfile('./static/data.csv'):
        with open('./static/data.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                user_id = int(row[0])
                if user_id > max_id:
                    max_id = user_id
    return max_id