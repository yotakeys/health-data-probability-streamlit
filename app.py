import streamlit as st
import numpy as np
import pandas as pd


class App():

    data = pd.DataFrame()
    data_attributes = list()
    data_form = dict()

    def init_app(self):
        st.title("Analisis")
        st.sidebar.title("Input Your Data")

    def get_data_excel(self):
        uploaded_file = st.sidebar.file_uploader("Choose a excel file")
        if uploaded_file is not None:

            self.data = pd.read_excel(uploaded_file)
            self.data_attributes = self.data.columns

            self.show_data_dataframe()
            self.show_form()

    def show_data_dataframe(self):
        col1, col2 = st.columns(2)
        col1.text("Data :")
        col1.write(self.data)

        col2.text("Data Attributes :")
        col2.write(self.data_attributes)

    def show_form(self):

        with st.sidebar.form("data_form"):

            for attr in self.data_attributes:
                if self.data[attr].dtype == 'int64':
                    self.data_form[attr] = st.number_input(
                        attr, value=0, step=1)

                elif self.data[attr].dtype == 'float64':
                    self.data_form[attr] = st.number_input(
                        attr, value=0, step=0.01)

                else:
                    self.data_form[attr] = st.selectbox(
                        attr, self.data[attr].unique())

            submitted = st.form_submit_button("Submit")

            if (submitted):
                self.show_data_input()

    def show_data_input(self):

        st.title("Data Input Value :")
        for key, value in self.data_form.items():
            st.write(key, ":", value)


app = App()
app.init_app()
app.get_data_excel()
