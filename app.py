import streamlit as st
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines import NelsonAalenFitter
import matplotlib.pyplot as plt
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter
from itertools import combinations


class App():

    data = pd.DataFrame()
    data_attributes = list()
    data_form = dict()
    submitted_key_column = False
    submitted_form_data = False
    file_is_uploaded = False
    kmf = KaplanMeierFitter()
    naf = NelsonAalenFitter()
    cph = CoxPHFitter()
    data_key_columns = {
        "duration": None,
        "event_observed": None
    }

    def init_app(self):
        st.title("Analisis")
        st.sidebar.title("Input Your Data")

    def get_data_excel(self):
        uploaded_file = st.sidebar.file_uploader("Choose a excel/csv file")
        if uploaded_file is not None and uploaded_file.name.endswith(".xlsx"):

            self.data = pd.read_excel(uploaded_file)

            self.data = self.data.dropna()
            single_unique_cols = self.data.columns[self.data.nunique() == 1]
            self.data = self.data.drop(single_unique_cols, axis=1)

            self.data_attributes = self.data.columns
            self.file_is_uploaded = True

            self.show_data_dataframe()
            self.get_data_key_column()
            # self.show_form()

        elif uploaded_file is not None and uploaded_file.name.endswith(".csv"):
            self.data = pd.read_csv(uploaded_file)

            self.data = self.data.dropna()
            single_unique_cols = self.data.columns[self.data.nunique() == 1]
            self.data = self.data.drop(single_unique_cols, axis=1)

            self.data_attributes = self.data.columns
            self.file_is_uploaded = True

            self.show_data_dataframe()
            self.get_data_key_column()
            # self.show_form()

    def show_data_dataframe(self):
        if self.file_is_uploaded:
            col1, col2 = st.columns(2)
            col1.text("Data :")
            col1.dataframe(self.data)

            col2.text("Data Attributes :")
            col2.dataframe(self.data_attributes)

    def get_data_key_column(self):
        if self.file_is_uploaded:
            with st.sidebar.form("data_key_column"):
                self.data_key_columns["duration"] = st.selectbox(
                    "Duration", self.data_attributes)
                self.data_key_columns["event_observed"] = st.selectbox(
                    "Event Observed", self.data_attributes)
                self.submitted_key_column = st.form_submit_button("Submit")

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

            self.submitted_form_data = st.form_submit_button("Submit")

            if self.submitted_form_data:
                self.add_data_input()
                self.show_data_input()

    def add_data_input(self):

        self.data = pd.concat([self.data, pd.DataFrame(
            self.data_form, index=[0])], ignore_index=True)

    def show_data_input(self):
        st.title("Data Input Value :")
        for key, value in self.data_form.items():
            st.write(key, ":", value)

    def show_analisis(self):
        if self.submitted_key_column:

            self.data[self.data_key_columns["event_observed"]
                      ] = self.data[self.data_key_columns["event_observed"]].astype('object')
            # Kurva Kaplan-Meier
            st.title("All Attributes")

            st.text("Kurva Kaplan-Meier")
            self.kmf.fit(durations=self.data[self.data_key_columns["duration"]],
                         event_observed=self.data[self.data_key_columns["event_observed"]], label="All Attributes")
            plt.figure(figsize=(10, 6))
            plt.title("Kurva Kaplan-Meier")
            plt.xlabel("Waktu (Bulan)")
            plt.ylabel("Survival Probability")
            self.kmf.plot_survival_function()
            st.pyplot(plt)

            # Cumulative Density
            plt.figure(figsize=(10, 6))
            plt.title("Cumulative Density")
            plt.xlabel("Waktu")
            plt.ylabel("Death Cumulative Probability")
            self.kmf.fit(durations=self.data[self.data_key_columns["duration"]],
                         event_observed=self.data[self.data_key_columns["event_observed"]], label="All attributes")
            self.kmf.plot_cumulative_density()

            st.text("Cummulative Density ")
            st.pyplot(plt)

            # Median Time to Event
            plt.figure(figsize=(10, 6))
            plt.title("Median Time to Event")
            plt.xlabel("Waktu")
            plt.ylabel("Conditional median time to event")
            self.kmf.fit(durations=self.data[self.data_key_columns["duration"]],
                         event_observed=self.data[self.data_key_columns["event_observed"]], label="All attributes")
            plt.plot(self.kmf.conditional_time_to_event_,
                     label="All attributes")

            st.text("Median Time to Event ")
            st.pyplot(plt)

            # Cumulative Hazard
            plt.figure(figsize=(10, 6))
            plt.title("Cumulative Hazard")
            plt.xlabel("Waktu")
            plt.ylabel("Deatch Cumulative Probability")
            self.naf.fit(durations=self.data[self.data_key_columns["duration"]],
                         event_observed=self.data[self.data_key_columns["event_observed"]], label="All attributes")
            self.naf.plot_cumulative_hazard()

            st.text("Cumulative Hazard")
            st.pyplot(plt)

            # Median Survival Time
            st.text("Median Survival Time : " +
                    self.kmf.median_survival_time_.__str__())

            st.text("Survival Function :")
            st.write(self.kmf.survival_function_)

            # Cox Proportional Hazard
            data_cox = self.data.select_dtypes(include=['int64', 'float64'])
            data_cox[self.data_key_columns["event_observed"]
                     ] = self.data[self.data_key_columns["event_observed"]]

            self.cph.fit(data_cox, duration_col=self.data_key_columns["duration"],
                         event_col=self.data_key_columns["event_observed"])
            st.text("Cox Proportional Hazard Summary")
            st.write(self.cph.summary)

            # Cox Proportional Hazard Plot
            plt.figure(figsize=(10, 6))
            plt.title("Cox Proportional Hazard")
            self.cph.plot()
            st.text("Cox Proportional Hazard Plot")
            st.pyplot(plt)

            # Per Column Uniqeu value
            for key in self.data_attributes:
                if key == self.data_key_columns["duration"] or key == self.data_key_columns["event_observed"]:
                    continue

                if self.data[key].dtype == 'int64' or self.data[key].dtype == 'float64':
                    continue

                st.title(key)
                self.show_median_survival_function(key)
                self.show_kurva_kaplan_meier(key)
                self.show_density_cumulative(key)
                self.show_conditional_time_event(key)
                self.show_cumulative_hazard(key)
                self.show_logrank_test(key)

    def show_kurva_kaplan_meier(self, key):

        plt.figure(figsize=(10, 6))
        plt.title("Kurva Kaplan-Meier")
        plt.xlabel("Waktu")
        plt.ylabel("Survival Probability")

        for attributes, attributes_df in self.data.groupby(key):
            self.kmf.fit(durations=attributes_df[self.data_key_columns["duration"]],
                         event_observed=attributes_df[self.data_key_columns["event_observed"]], label=attributes)
            self.kmf.plot_survival_function()

        st.text("Kurva Kaplan-Meier " + key)
        st.pyplot(plt)

    def show_median_survival_function(self, key):
        for attributes, attributes_df in self.data.groupby(key):
            self.kmf.fit(durations=attributes_df[self.data_key_columns["duration"]],
                         event_observed=attributes_df[self.data_key_columns["event_observed"]], label=attributes)
            st.text("Median Survival Time : " + key + " " + str(attributes) +
                    " : " + self.kmf.median_survival_time_.__str__())

    def show_density_cumulative(self, key):
        plt.figure(figsize=(10, 6))
        plt.title("Cummulative Density")
        plt.xlabel("Waktu")
        plt.ylabel("Death Probability")

        for attributes, attributes_df in self.data.groupby(key):
            self.kmf.fit(durations=attributes_df[self.data_key_columns["duration"]],
                         event_observed=attributes_df[self.data_key_columns["event_observed"]], label=attributes)
            self.kmf.plot_cumulative_density()

        st.text("Cummulative Density " + key)
        st.pyplot(plt)

    def show_conditional_time_event(self, key):
        plt.figure(figsize=(10, 6))
        plt.title("Median Time to Event")
        plt.xlabel("Waktu")
        plt.ylabel("Conditional median time to event")

        for attributes, attributes_df in self.data.groupby(key):
            self.kmf.fit(durations=attributes_df[self.data_key_columns["duration"]],
                         event_observed=attributes_df[self.data_key_columns["event_observed"]], label=attributes)
            plt.plot(self.kmf.conditional_time_to_event_, label=attributes)

        st.text("Median Time to Event " + key)
        st.pyplot(plt)

    def show_cumulative_hazard(self, key):

        plt.figure(figsize=(10, 6))
        plt.title("Cumulative Hazard")
        plt.xlabel("Waktu")
        plt.ylabel("Deatch Cumulative Probability")
        for attributes, attributes_df in self.data.groupby(key):
            self.naf.fit(durations=attributes_df[self.data_key_columns["duration"]],
                         event_observed=attributes_df[self.data_key_columns["event_observed"]], label=attributes)
            self.naf.plot_cumulative_hazard()

        st.text("Cumulative Hazard " + key)
        st.pyplot(plt)

    def show_logrank_test(self, key):
        combination_value = list(combinations(self.data[key].unique(), 2))
        for combi in combination_value:
            st.text("Logrank Test " + combi[0] + " and " + combi[1])
            st.write(logrank_test(self.data[self.data[key] == combi[0]][self.data_key_columns["duration"]],
                                  self.data[self.data[key] == combi[1]
                                            ][self.data_key_columns["duration"]],
                                  self.data[self.data[key] == combi[0]
                                            ][self.data_key_columns["event_observed"]],
                                  self.data[self.data[key] == combi[1]][self.data_key_columns["event_observed"]]))


app = App()
app.init_app()
app.get_data_excel()
app.show_analisis()
