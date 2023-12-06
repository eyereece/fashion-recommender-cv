<h2 align="center">SmartStylist: A Fashion Recommender System powered by Computer Vision</h2>
<br>

<center>
<a href="https://www.joankusuma.com/post/smart-stylist-a-fashion-recommender-system-powered-by-computer-vision"><img src='https://img.shields.io/badge/Project_Page-SmartStylist-pink' alt='Project Page'></a> 
<a href='https://www.joankusuma.com/post/object-detection-model-yolov5-on-fashion-images'><img src='https://img.shields.io/badge/Project_Page-ObjectDetection-blue' alt='Object Detection'></a> 
<a href='https://www.joankusuma.com/post/powering-visual-search-with-image-embedding'><img src='https://img.shields.io/badge/Project_Page-VisualSearch-green'></a> 
<a href='https://smartstylist.streamlit.app'><img src='https://img.shields.io/badge/Streamlit-Demo-red'></a>
</center>
<br>
<br>
<figure>
    <center>
        <img src="https://static.wixstatic.com/media/81114d_7f499b8207b848bc8bccfe1035a28b3d~mv2.png" alt="flowchart" height="350" width="600">
    </center>
</figure>

# Technical Features
* <b>Object Detection Model:</b> Leveraged the power of the YOLOv5 model trained on fashion images to detect fashion objects in images
* <b>Feature Extraction:</b> Utilized a Convolutional AutoEncoder implemented with PyTorch to extract latent features from detected fashion objects
* <b>Similarity Search Index: </b> Implemented FAISS library to construct an index, facilitating the search for visually similar outfits based on their distinct attributes

#### For more information on object detection model and feature extraction process, check out my repositories here:
* https://github.com/eyereece/yolo-object-detection-fashion
* https://github.com/eyereece/visual-search-with-image-embedding

<br>

# Project Demo

#### Online Streamlit Demo:
Try the [online streamlit demo](https://smartstylist.streamlit.app).

<b>Homepage:</b>

<figure>
    <center>
        <img src="https://static.wixstatic.com/media/81114d_e21c115d1ce141388a4ffc3ecd31c8ad~mv2.gif" alt="preview">
    </center>
</figure>

<br>

<b>Gallery:</b>

<figure>
    <center>
        <img src="https://static.wixstatic.com/media/81114d_47ce716d2b794785bb3b1b467b2ad425~mv2.gif" alt="preview">
    </center>
</figure>

<br>

<b>Object Detection Model: </b>

<figure>
    <center>
        <img src="https://static.wixstatic.com/media/81114d_f36652e9b7e844869ebb086e5f790beb~mv2.gif" alt="preview" height="500" width="500">
    </center>
</figure>

<br>

# Getting Started

Clone the repository: 
```bash
git clone https://github.com/eyereece/fashion-recommender-cv.git
```

Navigate to the project directory:
```bash
cd fashion-recommender-cv
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the streamlit app:
```bash
streamlit run home.py
```

<br>

# Usage
* Upload an image of an outfit (background in white works best)
* It currently only accepts jpg and png file
* Click "Show Recommendations" button to retrieve recommendations
* To update results, simply click on the "Show Recommendations" button again
* Navigate over to the sidebar, at the "gallery", to explore sample results

<br>

# Dataset

#### The dataset used in this project is available <a href="https://github.com/eileenforwhat/complete-the-look-dataset/tree/master">here</a>:
<div class="box">
  <pre>
    @online{Eileen2020,
  author       = {Eileen Li, Eric Kim, Andrew Zhai, Josh Beal, Kunlong Gu},
  title        = {Bootstrapping Complete The Look at Pinterest},
  year         = {2020}
}
  </pre>
</div>