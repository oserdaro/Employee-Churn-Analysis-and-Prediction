import streamlit as st
import pickle
import pandas as pd
from PIL import Image


# page title
st.title(":orange[Employee Churn Analysis and Prediction with XGBoost]")
st.subheader("Use the sidebar menu to change prediction parameters")

# Html adjustments to display the front end aspects
#html_temp = """
#<div style="background-color:orange;padding:10px">
#<h2 style="color:white;text-align:center;">Employee Churn Prediction </h2>
#</div>
#"""
#st.markdown(html_temp, unsafe_allow_html=True)

# Main Image
image = Image.open("image/office.jpg")
#st.image(image, use_column_width=True)

# Sidebar title
st.sidebar.title('Employee Churn Prediction')

# Sidebar image
st.sidebar.image(image, use_column_width=True)

# Side bar user inputs
department=st.sidebar.selectbox("Department", ('Sales', 'Technical', 'Support', 'IT', 'R&D', 'Product Management', 'Marketing', 'Accounting', 'HR', 'Management'))
salary=st.sidebar.radio('Salary',('Low','Medium','High'))
promotion=st.sidebar.radio("Promotion last 5 years:",('Yes','No'))
time=st.sidebar.slider("Length of service", 1, 10, step=1)
project=st.sidebar.slider("Number of projects", 1,10, step=1)
hours=st.sidebar.slider("Average monthly working hours", 80,400, step=10)
accident=st.sidebar.radio("Accident", ('Yes','No'))
satisfaction=st.sidebar.slider("Satisfaction score", 0.1,1.0, step=0.1)
evaluation=st.sidebar.slider("Last evaluation score", 0.1,1.0, step=0.1)

if promotion == "Yes":
    promotion = 1
else:
    promotion = 0

if accident == "Yes":
    accident = 1
else:
    accident = 0

# Converting user inputs to dataframe 
my_dict = {"satisfaction_level": satisfaction,
           "last_evaluation": evaluation,
           "number_project": project,
           'average_montly_hours': hours,
           'time_spend_company': time,
           "work_accident": accident,
           "promotion_last_5years": promotion,
           "departments": department,
           "salary": salary
}
df2 = pd.DataFrame.from_dict([my_dict])
df2.index = [''] * df2.shape[0]


st.subheader("You selected the following configuration:")
st.table(df2)


# Loading the model(s) to make predictions
loaded_model=pickle.load(open("xgb_model_with_transformer","rb"))
#transformer = pickle.load(open('pipeline_model.pkl', 'rb'))
#df3 = transformer.transform(df2)

# defining the function which will make the prediction using the data
def get_prediction(model, input_data):
	prediction = model.predict(input_data)
	return prediction


st.subheader("Press the 'Predict' button below to get a prediction")

if st.button("Predict"):
    result = get_prediction(loaded_model, df2)[0]
    if result == 0:
        result = "stay"
        st.success(f"The employee is likely to **{result}**")
        st.image(Image.open("image/lovemyjob.jpg"))
    elif result == 1:
        result = "leave"
        st.success(f"The employee is likely to **{result}**")
        st.image(Image.open("image/quit.jpg"))


