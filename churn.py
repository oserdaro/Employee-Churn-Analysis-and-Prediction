import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
from PIL import Image

# page title
st.title(":orange[Employee Churn Analysis and Churn Prediction with XGBoost]")

df0 = pd.read_csv('HR_Dataset.csv')
df = df0.copy()
df.drop("left", axis=1, inplace=True)

st.subheader("⚡ Select the type of visual from the dropdown below")

queries = ["[ satisfaction scores ]",
           "[ last evaluation scores ]",
           "[ number of projects they participated ]",
           "[ average monthly hours they worked ]",
           "[ time of service in the company ]",
           "[ whether they had work acciden ]",
           "[ promotion they received in the last 5 years ]",
           "[ department they worked in ]", 
           "[ salary category ]"
           ]
selection = st.selectbox("   ", options=queries  )
selected = queries.index(selection)
selected_column = df.columns[selected]
department_left_counts = df0.groupby([selected_column , 'left']).size().reset_index(name='count')
fig = px.bar(department_left_counts, x=selected_column, y='count', color='left', barmode='relative')
fig.update_layout(coloraxis_showscale=False)
st.success("◽ Showing the visual for : **Number of employees left and retained by "+selected_column+ "**")
st.plotly_chart(fig, use_container_width=True)


st.subheader("⚡ Use the sidebar menu to change the prediction parameters")

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


st.success("◽ Current prediction parameters are:")
#properties = {"border": "1px solid orange", "color": "green", "font-size": "12px", "text-align": "center"}
#st.table(df2.style.set_properties(**properties))
st.table(df2)

# Loading the model(s) to make predictions
loaded_model=pickle.load(open("xgb_model_with_transformer","rb"))
#transformer = pickle.load(open('pipeline_model.pkl', 'rb'))
#df3 = transformer.transform(df2)

# defining the function which will make the prediction using the data
def get_prediction(model, input_data):
	prediction = model.predict(input_data)
	return prediction


st.subheader("⚡ Press the 'Predict' button below to get a prediction")

if st.button("Predict"):
    result = get_prediction(loaded_model, df2)[0]
    if result == 0:
        result = "stay"
        st.success(f"◽ My prediction is :   **The employee is likely to {result}**")
        st.image(Image.open("image/lovemyjob.jpg"))
    elif result == 1:
        result = "leave"
        st.success(f"◽ My prediction is :   **The employee is likely to {result}**")
        st.image(Image.open("image/quit.jpg"))
