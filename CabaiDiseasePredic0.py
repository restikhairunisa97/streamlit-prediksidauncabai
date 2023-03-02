import streamlit as st
import numpy as np
from PIL import Image
from sklearnex.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearnex.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearnex.neighbors import KNeighborsClassifier
from sklearnex.ensemble import RandomForestClassifier
from sklearn import tree
#from GLCM.graylevelcooccurencematrix import GrayLevelCooccurenceMatrix
from LBP.localbinarypatterns import LocalBinaryPatterns
from sklearn.model_selection import KFold
import os
import cv2
import pandas as pd


def main_loop():
    # --- Tampilan header ---
    st.title("Prediksi Daun Cabai")
    st.subheader(
        "Prediksi Penyakit Pada Tanaman Cabai Berdasarkan Citra Daun Mengunakan Metode. "
        "Local Binary Pattern (LBP) Dan Support Vector Machine (SVM).")

    image_sehat = Image.open('dataset/imagesdataset/daun_sehat/health3.jpg')
    image_lubang = Image.open('dataset/imagesdataset/daun_berlubang/hole11.jpg')
    image_keriting = Image.open('dataset/imagesdataset/daun_keriting/curl3.jpg')
    image_bercak = Image.open('dataset/imagesdataset/daun_bercak/bercak4.jpg')
    st.image([image_sehat, image_lubang, image_keriting,image_bercak], width=160,
             caption=['Daun Sehat', 'Daun Berlubang', 'Daun Keriting', 'Daun Bercak'])


    # --- Membaca imagesdataset, mengekstraksi fitur2nya dan membentuk datasetcabai training ---
    dataset_dir = "dataset"
    data = []  # berisi nilai2 GLCM hasil ekstraksi
    labels = []  # berisi label2
    feature_names = []  # berisi nama kolom fitur

    metode_ekstraksi = LocalBinaryPatterns(24, 8)
    # membuat nama kolom fitur
    for i in range(0, 26):
        feature_names.append("feature" + str(i))

    st.write("### Processing Create Dataset From Images Extraction Using")
    st.write(metode_ekstraksi.__class__)

    # membaca citra training dan mengekstraksi fitur2nya
    for folder in os.listdir(dataset_dir):
        for sub_folder in os.listdir(os.path.join(dataset_dir, folder)):
            sub_folder_files = os.listdir(os.path.join(dataset_dir, folder, sub_folder))
            len_sub_folder = len(sub_folder_files) - 1
            for i, filename in enumerate(sub_folder_files):
                img = cv2.imread(os.path.join(dataset_dir, folder, sub_folder, filename))
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                gray_resize = metode_ekstraksi.crop_resize_imutils(gray, 128)
                hist = metode_ekstraksi.describe(gray_resize)
                data.append(hist)
                labels.append(
                    metode_ekstraksi.normalize_label(os.path.splitext(sub_folder)[0]))  # label pake nama foldernya

    st.text("The sample of origin datasetcabai is :")
    data_df = pd.DataFrame(data=data, columns=feature_names)
    data_df['label'] = labels
    st.dataframe(data_df)  # tampilkan dataframe datasetcabai

    # ------- Melatih data training ke algoritma klasifikasi ------
    model = SVC(C=100.0, random_state=42)

    st.write("### Processing Train the Model and Fit the Dataset Using")
    model.fit(data, labels)
    st.success('Done!')

    # ------- Menghitung Akurasi Model menggunakan KFold Cross Validation -------
    st.write("### Calculate the Accuration Model Using K- Fold Cross Validation with k=10")
    kFoldValidation = KFold(10)  # set nilai k=10
    modelScore = cross_val_score(model, data, labels, cv=kFoldValidation)
    st.write("K-Fold Cross Validation score:", modelScore)  # score akurasi model sebanyak k
    st.write("Mean KFCV score:", np.mean(modelScore))  # score akurasi model ratas dari k

    # ------- Memprediksi Citra dari cabaiimagestestset ------
    st.write("### Processing Prediction Of Image")
    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
    if not image_file:
        return None

    original_image = Image.open(image_file)
    original_image = np.array(original_image)

    # membuat citra gray dari original image
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # membuat citra gray crop_resize_imutils dan mengkestraksi ciri citra nya
    gray_crop_resize_imultils = metode_ekstraksi.crop_resize_imutils(gray_image, 128)
    hist = metode_ekstraksi.describe(gray_crop_resize_imultils)

    # prediksi citra yang di input
    prediction = model.predict(hist.reshape(1, -1))

    # menampilkan image dan hasil prediksinya
    original_image_resize = metode_ekstraksi.crop_resize_imutils(original_image, 224)
    cv2.putText(original_image_resize, prediction[0], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    lebel_leave = {'daunsehat': 'Daun Sehat', 'daunberlubang': 'Daun Berlubang', 'daunkeriting': 'Daun Keriting', 'daunbercak': 'Daun Bercak'}
    st.image([original_image_resize], caption=lebel_leave[prediction[0]])
    st.write("###", "Prediksi:", lebel_leave[prediction[0]])

    if prediction[0] == 'daunsehat':
        st.write(" saran menaman tanaman cabai agar cepat berbuah dan tidak layu")
        st.write(" jika menanam dari benih ")
        st.write(" 1. tanam benih cabai ke dalam wadah yg berisi tanah merah yang sudah di beri pupuk sekam ")
        st.write(" 2. siram benih secukupnya hingga muncul tunas")
        st.write(" 3. jika sudah muncul tunas dan tanaman mulai tumbuh, pindahkan ke wadah yang lebih besar yang sudah diberikan tanah merah yang di campur pupuk sekam")

    elif prediction[0] == 'daunberlubang':
        st.write("Pengendalian Daun Berlubang / Perforated Leaves : ")
        st.write(
            "1. penyemprotan insektisida Turex WP dengan konsentrasi 0,25 – 0,5 g/liter bergantian dengan insektisida Direct 25ec dengan konsentrasi 0,4 cc/liter atau insentisida Raydok 28ec dengan konsentrasi 0,25-0,5 cc/liter sehari sebelum pindah tanam.")
        st.write("2. Untuk tindakan pengendalian dianjurkan menyemprot pada sore atau malam hari")
        st.write("3. pasang perangkap imago hama")
        st.write("4. menjaga kebersihan kebun")
        st.write("5. Siangi gulma pada selasar bedengan, parit atau lubang-lubang mulsa")

    elif prediction[0] == 'daunbercak':
        st.write("Pengendalian Daun Bercak /spot leaf : ")
        st.write("1. Sanitasi dengan cara memusnahkan dan atau sisa-sisa tanaman yang terinfeksi/terserang")
        st.write(
            "2. Menanam bibit yang bebas patogen pada lahan yang tidak terkontaminasi oleh patogen, baik dipersemaian maupun di lapangan")
        st.write("3. Perlakuan benih sebelum tanam")
        st.write("4. Perbaikan drainase")
        st.write(
            "5. Waktu tanam yang tepat adalah musim kemarau dengan irigasi yang baik dan pergiliran tanaman dengan tanaman non solanaceae")
        st.write(
            "6. Pengendalian kimia dapat dilakukan dengan fungisida secara bijaksana, efektif, terdaftar dan diijinkan oleh Menteri Pertanian, berpedoman pada peramalan cuaca dan populasi spora di lapangan")

    else:
        st.write("Pengendalian Daun keriting / curly leaves: ")
        st.write("1. Menggunakan tanaman perangkap seperti kenikir kuning")
        st.write("2. Menggunakan mulsa perak")
        st.write("3. Sanitasi lingkungan dan pemotongan bagian tanaman yang terserang thrips")
        st.write("4. Sanitasi dengan mengeradikasi bagian tanaman yang terserang kemudian dimusnahkan")
        st.write(
            "5. Penggunaan perangkap warna kuning sebanyak 40 buah per ha atau 2 buah per 500 m2 yang dipasang sejak tanaman berumur 2 minggu. Perangkap dapat dibuat dari potongan bambu yang dipasang plastik map warna kuning. Plastik diolesi dengan lem agar thrips yang tertarik menempel. Apabila plastik sudah penuh dengan thrips maka plastik perlu diganti")
        st.write(
            "6. Pemanfaatan musuh alami yang potensial untuk mengendalikan hama thrips, antara lain predator kumbang Coccinellidae, tungau, predator larva Chrysopidae, kepik Anthocoridae dan patogen Entomophthora sp")
        st.write(
            "7. Pestisida digunakan apabila populasi hama atau kerusakan tanaman telah mencapai ambang pengendalian (serangan mencapai lebih atau sama dengan 15% per tanaman contoh) atau cara-cara pengendalian lainnya tidak dapat menekan populasi hama")
        st.write(
            "8. Pengendalian dengan akarisida yang efektif, terdaftar dan diijinkan Menteri Pertanian dilakukan apabila ditemukan gejala kerusakan daun dan populasi tungau")

if __name__ == '__main__':
    main_loop()
