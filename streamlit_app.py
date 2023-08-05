import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

page_bg_img = '''
<style>
[data-testid="stAppViewContainer"]{
background-image: url("https://images.unsplash.com/photo-1553447977-754f9430685c?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1740&q=80");
background-size: cover;
}
[data-testid="stHeader"]{
background-color: rgba(0,0,0,0);
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)
st.title('Book Recommendation System')
pt=pd.read_pickle('pivot.pkl')
finalbooks=pd.read_pickle('finalbooks.pkl')
similarity_score = cosine_similarity(pt)
def recommend(book_name):
    # index fetch
    index = np.where(pt.index==book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_score[index])),key=lambda x:x[1],reverse=True)[1:7]
    
    data = []
    for i in similar_items:
        item = []
        temp_df = finalbooks[finalbooks['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        
        data.append(item)
    
    return data
input_book = st.selectbox("Enter or select the name of book",pt.index)

if st.button('Recommend similar Books'):
    data_1=recommend(input_book)
    for i in data_1:
        st.subheader(i[0])
        st.write('Author : ', i[1])
        st.image(i[2])
        st.write("**______________________________________________________________________**")
    
