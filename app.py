import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import pandas as pd
from torch import mps
import supervision as sv

import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "sam_vit_h_4b8939.pth"
# sam_checkpoint = "sam_hq_vit_h.pth"
# sam_checkpoint = "sam_hq_vit_tiny.pth"

bg_image = '/Users/mayank/Documents/POCS/Segmentation App/image_10.png'

im = Image.open(bg_image) if bg_image else None
h = im.height if bg_image else None
w = im.width if bg_image else None

st.set_page_config(layout="wide")

st.write("height:",h)
st.write("width:",w)

model_type = "vit_h"
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
# device = "cpu"
st.write(device, mps.current_allocated_memory())
mps.set_per_process_memory_fraction(2.0)
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# def show_points(coords, labels, ax, marker_size=375):
#     pos_points = coords[labels==1]
#     neg_points = coords[labels==0]
#     ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
#     ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

# st.image('/Users/mayank/Documents/Object Detection/POCS/Segmentation App/output.png')

drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("point", "rect")
)
if drawing_mode == 'point':
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)

stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
# bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
# bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
update = st.sidebar.button("Get segmentations")

canvas_result = st_canvas(
    fill_color= None,  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    height = h,
    width = w,
    background_image=im if bg_image else None,
    update_streamlit = update,
    drawing_mode=drawing_mode,
    point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
    key="canvas",

)
# canvas.draw_image('/Users/mayank/Documents/Object Detection/POCS/Segmentation App/output.png', 50, 50)

# def handle_mouse_down(x, y):
#     # Do something else
#     pass
# canvas.on_mouse_down(handle_mouse_down)

if canvas_result.image_data is not None:
    st.image(canvas_result.image_data)
if canvas_result.json_data is not None:
    objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
    for col in objects.select_dtypes(include=['object']).columns:
        objects[col] = objects[col].astype("str")
    st.dataframe(objects)
    box = []
    for i in range (len(objects["left"])):
        box.append([objects["left"][i],objects["top"][i],objects["left"][i]+objects["width"][i],objects["top"][i]+objects["height"][i]])
    input_boxes =  torch.tensor(box, dtype = torch.float32 ,device=predictor.device) if bg_image else None

    image = cv2.imread(bg_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    # from streamlit_image_coordinates import streamlit_image_coordinates

    # value = streamlit_image_coordinates(Image.open("/Users/mayank/Documents/Object Detection/POCS/Segmentation App/coco_train_cars/images/05135aca-Lamborghini_train10.jpg"),
    #             key="pil",
    #             )
    # st.write(value)

    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
    results = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
        return_logits= True,
    )
    # st.write(masks.cpu().numpy())
    # st.write(scores.cpu().numpy())
    # st.write(logits.cpu().numpy())
    
    # result = [{}]
    st.set_option('deprecation.showPyplotGlobalUse', False)

    plt.imshow(image)
    for mask in results[0]:
        # st.write( mask.cpu().numpy().shape[-2:])
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
    for box in input_boxes:
        show_box(box.cpu().numpy(), plt.gca())
    # plt.axis('off')
    # plt.show()
    st.pyplot(plt.show())
    detections = sv.Detections.from_detectron2(results)
    polygons = [sv.mask_to_polygons(m.cpu().numpy()) for m in detections.mask]
    st.write(polygons)