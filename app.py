import streamlit as st 
import keras

import numpy as np
from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer

model=keras.models.load_model('ml_final2_model.hdt5')
        

st.title("Detecting Depression Level Using Social Media Posts")
	# st.subheader("ML App with Streamlit")
html_temp = """
	<div style="background-color:green;padding:10px">
	<h1 style="color:white;text-align:center;">Depressioin Detecting  App </h1>
	</div>

	"""
st.markdown(html_temp,unsafe_allow_html=True)
post_text = st.text_area("TYPE YOUR POST HERE","")

if st.button("Predict"):
    max_words = 4000
    max_len = 400
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(post_text)
    sequences = tokenizer.texts_to_sequences(post_text)
    post_text = pad_sequences(sequences, maxlen=max_len)
    test_prediction =model.predict(post_text)
    if np.around(test_prediction, decimals=0)[0][0] == 1.0:
        st.write('You are depressed.Please visit the counselor')
    else:
        st.write("You are not depressed")

        