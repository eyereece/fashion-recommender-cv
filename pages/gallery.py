import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from src.utilities import viz_thumbnail

st.set_page_config(page_title="Gallery",
                   page_icon=":shopping_bags:")

st.markdown("# :female_fairy: :frame_with_picture:")
st.markdown("# :rainbow[App Gallery] :magic_wand:")

# Load your images
# pink-white
pw_path = 'gallery/sample_query/pink-white/pw_1.jpg'

# black-coat
bc_path = 'gallery/sample_query/black-coat/bc_1.jpg'

# sweater-skirt
ss_path = 'gallery/sample_query/sweater-skirt/ss_1.jpg'

# black-jacket
bk_path = 'gallery/sample_query/black-jacket/bk_1.jpg'

#Set the size for thumbnail
thumbnail_size = (10, 10)

tab1, tab2, tab3, tab4 = st.tabs(['Outfit 1','Outfit 2','Outfit 3', 'Outfit 4'])

with tab1:
    st.markdown("#### :rainbow[Query Image:]")
    fig1 = viz_thumbnail(pw_path, thumbnail_size)
    st.pyplot(fig1)
    st.divider()
    st.markdown("#### :rainbow[Recommendations:]")
    st.image('gallery/sample_results/pink-white/pw_im1.png')
    st.image('gallery/sample_results/pink-white/pw_im2.png')

with tab2:
    st.markdown("#### :rainbow[Query Image:]")
    fig2 = viz_thumbnail(bc_path, thumbnail_size)
    st.pyplot(fig2)
    st.divider()
    st.markdown("#### :rainbow[Recommendations:]")
    st.image('gallery/sample_results/black-coat/bc_im1.png')
    st.image('gallery/sample_results/black-coat/bc_im2.png')
    
with tab3:
    st.markdown("#### :rainbow[Query Image:]")
    fig3 = viz_thumbnail(ss_path, thumbnail_size)
    st.pyplot(fig3)
    st.divider()
    st.markdown("#### :rainbow[Recommendations:]")
    st.image('gallery/sample_results/sweater/ss_im1.png')
    st.image('gallery/sample_results/sweater/ss_im2.png')

with tab4:
    st.markdown("#### :rainbow[Query Image:]")
    fig4 = viz_thumbnail(bk_path, thumbnail_size)
    st.pyplot(fig4)
    st.divider()
    st.markdown("#### :rainbow[Recommendations:]")
    st.image('gallery/sample_results/black-jacket/bk_im1.png')
    st.image('gallery/sample_results/black-jacket/bk_im2.png')