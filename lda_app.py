import streamlit as st
import nltk
import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import streamlit.components.v1 as components


nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

st.set_page_config(page_title="Topic Explorer", layout="wide")
st.title(" LDA Topic Explorer")
st.markdown("Upload a `.txt` file or paste text below to discover hidden topics!")


st.sidebar.header("Settings")
num_topics = st.sidebar.slider("Number of Topics", 2, 10, 3)


uploaded_file = st.file_uploader(" Upload a .txt file", type=["txt"])
if uploaded_file is not None:
    user_text = uploaded_file.read().decode("utf-8")
else:
    user_text = st.text_area("Or paste your text here:", height=300)


def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return tokens


def plot_wordcloud(words):
    wc = WordCloud(background_color="white", max_words=100).generate(words)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)


if st.button(" Analyze Topics"):
    if not user_text.strip():
        st.warning("Please provide some text.")
    else:
        with st.spinner("Running LDA..."):
            texts = [preprocess(user_text)]
            dictionary = corpora.Dictionary(texts)
            corpus = [dictionary.doc2bow(text) for text in texts]
            lda_model = gensim.models.LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=num_topics,
                passes=15,
                random_state=42
            )

        st.success("Topic modeling complete!")
        st.subheader(f" Top {num_topics} Topics")

        for idx, topic in lda_model.print_topics(-1):
            words = topic.split('+')
            word_list = [w.split('*')[1].replace('"', '').strip() for w in words]
            st.markdown(f"### 🔹 Topic {idx+1}")
            st.write(", ".join(word_list))
            plot_wordcloud(" ".join(word_list))

       
        st.subheader(" Interactive Topic Visualization (PyLDAvis)")
        with st.spinner("Generating visualization..."):
            vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
            html_string = pyLDAvis.prepared_data_to_html(vis_data)
            components.html(html_string, height=800, scrolling=True)
