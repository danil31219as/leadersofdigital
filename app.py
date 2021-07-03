import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import json
import pickle, gdown
import torch
import io
import numpy as np
import gdown
from classes import (
    Data, Cutter, ResNet, BearCatcher, Predictor,
    draw_bboxes, normalize, device
)

drive_link = "https://drive.google.com/u/0/uc?id=1-1D2HLYzcg8yCTB_vKX3_P0MoLJvwAer"


@st.cache
def download_model():
    gdown.cached_download(drive_link, "BearCatcher.pkl", quiet=False)


@st.cache
def read_files(files):
    imgs = list()
    for file in files:
        try:
            imgs.append(file.name)
            with open(file.name, 'wb') as f:
                f.write(file.getvalue())
        except:
            imgs.append((file.name, None))
    return imgs


@st.cache
def read_stats(file):
    with open(file, 'r') as f:
        stats = json.loads(f.read())
        try:
            recall = stats['recall'][-1]
            precision = stats['precision'][-1]
        except:
            recall, precision = 1, 0.951

    return recall, precision, stats['losses']['train'], stats['losses']['val'] # изменить лоссы и iou


def main():
    download_model()
    st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background: url("https://i.imgur.com/0ONDq1e.jpeg");
        background-repeat: repeat;
        background-size: 50% auto;
    }
    .reportview-container {
        background: url("https://i.imgur.com/0ONDq1e.jpeg");
        background-repeat: repeat;
        background-size: 50% auto;    }
    </style>
    """, unsafe_allow_html=True)
    st.title('Polar Bear Detector')

    st.write("""
    Детекция полярных медведей на арктических снимках **by Machine Brain**
    """)

    with st.beta_expander("Информация о модели"):
        recall, precision, train_loss, val_loss = read_stats('./statistics/resnet_stat.json')

        st.write("Точность по метрике *IoU*: **86.3%**")

        st.subheader('Схема обработки картинки')
        st.image('images/image_proc.png')

        st.subheader('Схема работы алгоритма')
        st.image('images/model_work.png')

        df = pd.DataFrame([[recall, precision]], columns=('Recall', 'Precision'))
        df.index=['']
        st.subheader('Полнота и точность')
        st.dataframe(df)

        st.subheader('Проверка на переобучение')
        fig, ax = plt.subplots()
        epochs = list(range(1,len(train_loss)+1))
        ax.plot(epochs, train_loss, label = "train loss")
        ax.plot(epochs, val_loss, label = "val loss")
        ax.legend()
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        st.pyplot(fig)

    st.subheader("Загрузка снимков")
    filenames = st.file_uploader('Выберите или ператащите сюда фотографии', type=['png','jpeg', 'jpg'], accept_multiple_files=True)

    if st.button("Загрузить") and filenames:
        #st.info('Файлы успешно загружены')
        images = read_files(filenames)  
        
        
        model = pickle.load(open('BearCatcher.pkl', 'rb'))
        predictor = Predictor(model, images)
        predicted_images, predicted_results = list(), list()

        for image in predictor:
            if image:
                img = image[0]
                preds = image[1]
                print(len(img))
                print(preds)
                predicted_images.append(img)
                predicted_results.append(preds)

        with st.beta_expander("Статистика"):
            st.write(f'Всего загружено фотографий: **{len(filenames)}**')

            labels = ('Фото с медведями', 'Пустые фото')
            bears = sum([1 for pred in predicted_results if pred != []])
            sizes = (bears,len(predicted_results) - bears)
            explode = (0.1, 0)
            colors = ['#8e90d4','#d1cfcf']

            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, colors=colors, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
            ax1.axis('equal')

            st.pyplot(fig1)

        with st.beta_expander("Информация о каждом фото"):
            for ind in range(len(predicted_images)):
                im  = np.array(Image.open(images[ind]))
                st.subheader(images[ind])
                st.write(f'Найдено медведей: **{len( predicted_results[ind])}**')
                print('BBB ', predicted_results[ind], images[ind])
                if len( predicted_results[ind]) != 0:
                    print('CCC')
                    print(im.shape)
                    st.image(draw_bboxes(im, predicted_results[ind]))
                else:
                    st.image(im)


if __name__ == '__main__':
    main()
