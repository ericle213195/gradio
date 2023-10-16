import gradio as gr
import utils as u
from face_detector import YoloV5FaceDetector
import tensorflow as tf
import os
import csv
port = int(os.environ.get('PORT', 8000))
# initialize a face detector
face_crop = YoloV5FaceDetector()
# initialize face recognition model
model_interf = "GhostFaceNet_W1.3_S1_ArcFace.h5"
if isinstance(model_interf, str) and model_interf.endswith("h5"):
    model = tf.keras.models.load_model(model_interf)
    model_interf = lambda imms: model((imms - 127.5) * 0.0078125).numpy()
else:
    model_interf = model_interf

    

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

#app 1
def add_new_person(name,img1,img2,img3):
    nid = get_max_user_id()+1
    imgs = [img1, img2, img3]
    imgs = [u.resize_image(imgg) for imgg in imgs]
    print(len(imgs))
    print("new ID: ",nid)
    a,_,_ = u.add_new_person(imgs,nid,model_interf,face_crop,name)
    if (a.shape[0]-1!=int(nid)):
        return 'ERROR: Something goes wrong with adding this new person, current embedding shape: '+str(a.shape)

    return "Successfully register your attendance!😎"

#app 2
def checkin(img):
  img = u.resize_image(img)
  image_id,image_list,distances,person_id  = u.return_id_imgs(img,model_interf,face_crop)
  person_name = u.find_string_by_person_id(person_id)
  return "Checked in: " + person_name

#interface 1
app1 =  gr.Interface(fn = add_new_person, inputs=["text",gr.Image(type="numpy"),gr.Image(type="numpy"),gr.Image(type="numpy")], outputs="text")
#interface 2

# place your code here
app2 =  gr.Interface(fn = checkin, inputs=gr.Image(type="numpy"), outputs="text")

demo = gr.TabbedInterface([app1, app2], ["Register Your Attendance", "Check Your Attendance"])

demo.launch(server_name="0.0.0.0", server_port=port, debug=True)