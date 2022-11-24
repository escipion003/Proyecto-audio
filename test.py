
import numpy as np
import streamlit as st
from audio_featurizer import audio_process, spectrogram_plot, model1, model2#,soundwaves_plot

def main():
    """music gender classifier app"""

    # Imagen de Fondo
    def add_bg_from_url():
        st.markdown(
            f"""
             <style>
             .stApp {{
                 background-image: url("https://i2.wp.com/highland-music.com/wp-content/uploads/2016/04/Blue-Background-Music-Headphone-Wallpaper-Picture-HD-Free-298292334-e1459743028815.png?ssl=1");
                 background-attachment: fixed;
                 background-size: cover
             }}
             </style>
             """,
            unsafe_allow_html=True
        )
    add_bg_from_url()



    # Titulo y subtitulo
    #st.title("Music Gender ML App")
    #st.subheader("Insert Audio")

    # Barra GRIS
    html_temp ="""
    <div style="background-color:grey;padding:15px;">
    <h2> Streamlit ML App</h2>
    </div>
    """

    st.markdown(html_temp,unsafe_allow_html=True)

    # My youtube channel link and sidebar
    st.sidebar.markdown(
        """<h1 style='text-align: center;color:  #0e76a8;'><a style='text-align: center;color:  #0e76a8;' href="https://www.youtube.com/watch?v=8M20LyCZDOY&ab_channel=DataProfessor" target="_blank">My Youtube Channel</a></h1>""",
        unsafe_allow_html=True)
    # st.sidebar.markdown("""<h1 style='text-align: center;color: black;' ><a style='text-align: center;color: black;'href="https://github.com/rohankokkula/TEATH" target="_blank">Github Source Code</a></h1>""", unsafe_allow_html=True)


    #
    # st.markdown(
    #     """<h1 style='text-align: center; color: white;font-size:60px;margin-top:-50px;'>AUDIO CLASSIFIER</h1><h1 style='text-align: center; color: white;font-size:30px;margin-top:-30px;'>Using Machine Learning</h1>""",
    #     unsafe_allow_html=True)

    
    

   ###################################################################################################################################################################


    check = st.sidebar.checkbox('Probar un demo')

    if check:
        rad_test = st.sidebar.radio("Select format of audio file", options=['mp3'])

        if rad_test == "mp3":
            rad_file = st.sidebar.radio("Select the name of song", ["Trumpet Tune"])
            if rad_file == "Trumpet Tune":
                rad = st.sidebar.radio("Choose Options", options=["Predict", "Spectrogram"])
                st.audio("C:/Users/USUARIO/PycharmProjects/pythonProject2/Trumpet Tune.mp3")
                # rad = st.sidebar.checkbox(label="Do You want to see the spectrogram ?")
                if rad == "Predict":
                    if st.button("Classify Audio"):
                        uploaded_audio = audio_process("C:/Users/USUARIO/PycharmProjects/pythonProject2/Trumpet Tune.mp3")





                        generos = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae",
                                   "rock"]

                        out_xgb = model2(uploaded_audio)[0]
                        predictions = generos[out_xgb]

                        st.markdown(
                            f"""<h1 style='color:yellow;'>Prediction xgb : <span style='color:white;'>{predictions}</span></h1>""",
                            unsafe_allow_html=True)

                elif rad == "Spectrogram":
                    fig = spectrogram_plot("Man Out Of Town.mp3")
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    st.markdown(
                        f"""<h1 style='color:yellow;'>Spectrogram : </h1>""",
                        unsafe_allow_html=True)
                    st.pyplot(fig)






    ##################################################################################################################





    # Seleccionar audio
    radio = st.sidebar.radio("Select format of audio file", options=['mp3', 'wav'])
    
    

    if radio == 'wav':

        file = st.sidebar.file_uploader("Upload Audio To Classify", type=["wav"])

        if file is not None:
            st.markdown(
                """<h1 style='color:white;'>Audio : </h1>""",
                unsafe_allow_html=True)
            st.audio(file)

            # Predecir
            rad = st.sidebar.radio("Choose model", options=["XGB","Rforest","Spectrogram"])

            #rad = st.sidebar.checkbox(label="Do You want to see the spectrogram ?")
            if rad == "XGB":
                if st.button("Classify Audio"):
                    uploaded_audio = audio_process(file)

                    #out_forest = rforest.predict(x_new)[0]

                    generos = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae",
                               "rock"]



                    out_xgb = model2(uploaded_audio)[0]
                    predictions = generos[out_xgb]

                    #out_forest = model1(uploaded_audio)[0]
                    #predictions2 = generos[out_forest]

                    #targets = encoding.inverse_transform(np.array(predictions).reshape(1, -1))
                    #
                    #st.write(predictions)
                    #
                    #
                    #st.success(predictions)

                    st.markdown(
                        f"""<h1 style='color:yellow;'>Prediction xgb : <span style='color:white;'>{predictions}</span></h1>""",
                        unsafe_allow_html=True)
                    # st.markdown(
                    #     f"""<h1 style='color:yellow;'>Prediction rforest : <span style='color:white;'>{predictions2}</span></h1>""",
                    #     unsafe_allow_html=True)
            elif rad == "Rforest":
                if st.button("Classify Audio"):
                    uploaded_audio = audio_process(file)

                    #out_forest = rforest.predict(x_new)[0]

                    generos = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae",
                               "rock"]



                    #out_xgb = model2(uploaded_audio)[0]
                    #predictions = generos[out_xgb]

                    out_forest = model1(uploaded_audio)[0]
                    predictions2 = generos[out_forest]

                    #targets = encoding.inverse_transform(np.array(predictions).reshape(1, -1))
                    #
                    #st.write(predictions)
                    #
                    #
                    #st.success(predictions)

                    # st.markdown(
                    #     f"""<h1 style='color:yellow;'>Prediction xgb : <span style='color:white;'>{predictions}</span></h1>""",
                    #     unsafe_allow_html=True)
                    st.markdown(
                        f"""<h1 style='color:yellow;'>Prediction rforest : <span style='color:white;'>{predictions2}</span></h1>""",
                        unsafe_allow_html=True)


            elif rad == "Spectrogram":
                fig = spectrogram_plot(file)
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.markdown(
                    f"""<h1 style='color:Green;'>Spectrogram : </h1>""",
                    unsafe_allow_html=True)
                st.pyplot(fig)

            # rad2 = st.sidebar.radio("Choose Visualization", options=[ "Spectrogram","Audio Waves"])
            #
            # if rad2 == "Spectrogram":
            #     fig = spectrogram_plot(file)
            #     st.set_option('deprecation.showPyplotGlobalUse', False)
            #     st.markdown(
            #         f"""<h1 style='color:White;'>Spectrogram : </h1>""",
            #         unsafe_allow_html=True)
            #     st.pyplot(fig)
            # elif rad2 == "Audio Waves":
            #     fig = soundwaves_plot(file)
            #     st.set_option('deprecation.showPyplotGlobalUse', False)
            #     st.markdown(
            #         f"""<h1 style='color:White;'>Audio Waves : </h1>""",
            #         unsafe_allow_html=True)
            #     st.pyplot(fig)
if __name__ == "__main__":
    main()

#C:/Users/USUARIO/PycharmProjects/pythonProject2/test.py
