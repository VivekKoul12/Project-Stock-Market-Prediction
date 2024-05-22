import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go
import datetime as dt

# Load ticker.csv file into a Pandas DataFrame

def load_data():
    df = pd.read_csv('ticker.csv')
    return df
df = load_data()

# Display a text input box for the user to enter the company name
user_input = st.text_input('Enter Company Name')

# Filter DataFrame based on user input to find matching company names
#contains matches the substring like if we write app then the compines started with app will be made into the df
matching_companies = df[df['Company Name'].str.contains(
    user_input, case=False)]

# Display the matching company names as recommendations
#here the data frame created based on input will be shown
if not matching_companies.empty:
    selected_company = st.selectbox(
        "Select a Company:", matching_companies['Company Name'])
    user_input = matching_companies.loc[matching_companies['Company Name']
                                        == selected_company, 'Ticker'].iloc[0]
    st.write(f"Selected Ticker: {user_input}")
else:
    st.write("No matching companies found.")

# Create a Plotly figure for the Closing Price vs Time chart
if st.button('Submit'):
    try:
        df = yf.download(user_input, start='2010-01-01', end=dt.datetime.now())

        # Check if the data size is less than 2 years (approximately 730 days)
        if len(df) < 730:
            raise ValueError(
                "Data size is less than 2 years.")
        fig = go.Figure()
        # Add trace for closing price
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'], mode='lines', name='Closing Price'))

        # Customize layout
        fig.update_layout(
            title='Closing Price vs Time',
            xaxis_title='Date',
            yaxis_title='Closing Price',
        )

        # Display the interactive Plotly chart
        st.plotly_chart(fig)


        # Calculate 100-day and 200-day Moving Averages
        ma100 = df['Close'].rolling(window=100).mean()
        ma200 = df['Close'].rolling(window=200).mean()

        # Create a Plotly figure for the Closing Price vs Time chart with 100MA & 200MA
        fig = go.Figure()

        # Add trace for 100-day Moving Average
        fig.add_trace(go.Scatter(x=df.index, y=ma100, mode='lines',
                                name='100-day Moving Average', line=dict(color='red')))

        # Add trace for 200-day Moving Average
        fig.add_trace(go.Scatter(x=df.index, y=ma200, mode='lines',
                                name='200-day Moving Average', line=dict(color='green')))

        # Add trace for closing price
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'], mode='lines', name='Closing Price', line=dict(color='blue')))

        # Customize layout
        fig.update_layout(
            title='Closing Price vs Time with 100-day & 200-day Moving Averages',
            xaxis_title='Date',
            yaxis_title='Price',
        )
        height = 600,  # Set the height of the figure
        width = 1000

        # Display the interactive Plotly chart
        st.plotly_chart(fig)

       

        # Customize layout
        fig.update_layout(
            title='Predictions vs Original',
            xaxis_title='Date',
            yaxis_title='Price',
        )
        height = 600,  # Set the height of the figure
        width = 1000

        # Display the interactive Plotly chart
        st.plotly_chart(fig)

        # # Function to predict stock prices for a whole week

        def predict_weekly_prices(model, scaler, past_data):
            future_predictions = []

            for _ in range(7):
                # Ensure the past_data is reshaped to 2D array (if not already)
                if len(past_data.shape) == 1:
                    past_data = past_data.reshape(-1, 1)
                # Scale the input data
                scaled_input_data = scaler.transform(
                    past_data[-300:])  # Increase window size to 200

                # Reshape input data for prediction
                x_test = scaled_input_data.reshape(1, 300, 1)  # Adjust window size

                # Make prediction
                predicted_price = model.predict(x_test)

                # Inverse scaling to get the actual price
                predicted_price = scaler.inverse_transform(predicted_price)

                # Append the predicted price to the list of future predictions
                future_predictions.append(predicted_price[0, 0])

                # Update input data for the next prediction
                past_data = np.append(past_data, predicted_price)

            return future_predictions

        # Load the model
        model = load_model('keras_model.h5')
        # Load past data (replace this with your actual past data)
        past_data = df[len(df)-300:len(df)]['Close']
        # Convert the Series to a DataFrame
        past_data_df = past_data.reset_index().tail(7)
        past_data_df.columns = ['Date', 'Close']

        # Display the last seven days of prices without the index
        st.subheader('Prices for last seven days')
        st.write(past_data_df.to_html(index=False), unsafe_allow_html=True)

        # Convert pandas Series to NumPy array
        past_data_array = past_data.to_numpy()

        # Recreate the scaler and fit it to past data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(past_data_array.reshape(-1, 1))  # Corrected the reshaping here

        # Predict stock prices for the next 7 days
        weekly_predictions = predict_weekly_prices(model, scaler, past_data_array)

        # Generate dates for the next 7 days
        last_date = df.index[-1]
        next_week_dates = pd.date_range(
            last_date + dt.timedelta(days=1), periods=7)

        # Combine dates and predicted prices into a DataFrame
        weekly_predictions_df = pd.DataFrame({
            'Date': next_week_dates,
            'Predicted Price': weekly_predictions
        })

        # Display the predicted prices for the next week with corresponding dates on Streamlit without index
        st.subheader('Predicted Prices for the Next Week')
        st.write(weekly_predictions_df.to_html(
            index=False), unsafe_allow_html=True)
    except ValueError as e:
        st.error(str(e))
