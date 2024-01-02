import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from obj_detection import ObjDetection
from PIL import Image
from torchvision import transforms

from src.utilities import ExactIndex, extract_img, similar_img_search, display_image, visualize_nearest_neighbors, visualize_outfits


# --- UI Configurations --- #
st.set_page_config(page_title="Smart Stylist powered by computer vision",
                   page_icon=":shopping_bags:"
                   )

st.markdown("# :female_fairy: :shopping_bags:")
st.markdown("# :rainbow[Your personal AI Stylist] :magic_wand:")

# --- Message --- #
st.write("Hello, welcome to my project page! :smiley:")
st.write("Smart Stylist is a computer vision powered web-app that lets you upload an image of an outfit and return recommendations on similar style. An image with white background works best. ")
st.write("For more information on how the system works, check out the project page [here](https://www.joankusuma.com/post/smart-stylist-a-fashion-recommender-system-powered-by-computer-vision) ")
st.divider()
st.info("Check out the gallery in sidebar to get some ideas", icon="👈🏼")

# --- Load Model and Data --- #
with st.spinner('Please wait while your model is loading'):
    yolo = ObjDetection(onnx_model='./models/best.onnx',
                        data_yaml='./models/data.yaml')
    
index_path = "flatIndex.index"

with open("img_paths.pkl", "rb") as im_file:
    image_paths = pickle.load(im_file)

with open("embeddings.pkl", "rb") as file:
    embeddings = pickle.load(file)

def load_index(embeddings, image_paths, index_path):
    loaded_idx = ExactIndex.load(embeddings, image_paths, index_path)
    return loaded_idx

loaded_idx = ExactIndex.load(embeddings, image_paths, index_path)

# --- Image Functions --- #
transformations = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

def upload_image():
    image_file = st.file_uploader(label='Upload Image')
    if image_file is not None:
        if image_file.type in ('image/png', 'image/jpeg'):
            st.success('Valid Image File Type')
            return image_file
        else:
            st.error('Only the following image files are supported (png, jpg, jpeg)')

# --- Object Detection and Recommendations --- #
def main():
    object = upload_image()

    if object:
        prediction = False
        image_obj = Image.open(object)
        st.image(image_obj)
        button = st.button('Show Recommendations')
        if button:
            with st.spinner(""" Detecting Fashion Objects from Image. Please Wait. """):
                image_array = np.array(image_obj)
                cropped_objs = yolo.crop_objects(image_array)
                if cropped_objs is not None:
                    prediction = True
                else:
                    st.caption("No fashion objects detected.")

        if prediction:
            cropped_objs = [obj for obj in cropped_objs if obj.size > 0]

            # The following comments visualized detected fashion objects
            # st.caption(":rainbow[Detected Fashion Objects]")  
            # if len(cropped_objs) == 1:
            #    st.image(cropped_objs[0])
            #else:
                # If there's more than one images
            #    fig, axes = plt.subplots(1, len(cropped_objs), figsize=(15, 3))
            #    for i, obj in enumerate(cropped_objs):
            #            axes[i].imshow(obj)
            #            axes[i].axis('off')         
            #    st.pyplot(fig)

            # st.caption(":rainbow[Recommended Items]")
            with st.spinner("Finding similar items ..."):
                boards = []
                for i, obj in enumerate(cropped_objs):
                    embedding = extract_img(obj, transformations)
                    selected_neighbor_paths = similar_img_search(embedding, loaded_idx)
                    boards.append(selected_neighbor_paths)

                # Flatten list of lists into a single list of paths
                all_boards = [path for sublist in boards for path in sublist]

                # Visualize recommended outfits
                rec_fig = visualize_outfits(all_boards)
                st.pyplot(rec_fig)

if __name__ == "__main__":
    main()