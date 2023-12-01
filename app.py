import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open("C:/Users/dalal/OneDrive/Desktop/Insurance Fraud/trained_model.sav","rb"))

scaler = pickle.load(open("C:/Users/dalal/OneDrive/Desktop/Insurance Fraud/standardized_1.pkl","rb"))


def Insurance(input_data):

#changing the input data into the numpy array data
    input_data_as_numpy_array = np.asarray(input_data)

#reshape the array as we are predicting for one instances
    input_data_reshape = input_data_as_numpy_array.reshape(1,-1)

#standization of input data
    std_data = scaler.transform(input_data_reshape)
    print(std_data)

    prediction = loaded_model.predict(std_data)
    print(prediction)

    if (prediction[0]==0):
         return "No Fraud"
    else:
        return "Fraud"


def main():

    st.title("Insurance Fraud Detection")#policy_deductable,umbrella_limit,capital-gains,capital-loss,incident_hour_of_the_day,number_of_vehicles_involved,bodily_injuries,witnesses,injury_claim,property_clai

    months_as_customer = st.text_input('Months')
    policy_deductable =  st.text_input('Policy Deductable')
    umbrella_limit =  st.text_input('Limit')
    capital_gains =  st.text_input('Profit')
    capital_loss =  st.text_input('Loss')
    incident_hour_of_the_day = st.text_input('Hour of the Day')
    number_of_vehicles_involved = st.text_input('No. Vehicles Involoved')
    bodily_injuries = st.text_input('Injuries')
    witnesses = st.text_input('Witnesses')
    injury_claim = st.text_input('Injury Cost claimed')
    property_claim = st.text_input('Property cost claimed')
    vehicle_claim = st.text_input("vehicle cost claimed")

    Result = ""

    if st.button("Fraud Result"):
        Result = Insurance([months_as_customer,policy_deductable,umbrella_limit,capital_gains,capital_loss,incident_hour_of_the_day,number_of_vehicles_involved,bodily_injuries,witnesses,injury_claim,property_claim,vehicle_claim])
    st.success(Result)


if __name__ == "__main__":
    main()