from flask import Flask, render_template, request, redirect, url_for
import pickle
from datetime import datetime

app = Flask(__name__)

# Load the scaler and model once, globally
with open('artifact\scaled_data.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('artifact/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predicter', methods=['POST', 'GET'])
def predicter():
    if request.method == 'POST':
        try:
            # Retrieve form data and convert to numeric types
            Outliers = float(request.form.get('Outliers', 0))                           
            PricingStrategy = float(request.form.get('PricingStrategy', 0))                   
            ProductCategory_airtime  = float(request.form.get('ProductCategory_airtime', 0))         
            ProductCategory_financial_services  = float(request.form.get('ProductCategory_financial_services', 0)) 
            Value = float(request.form.get('Value', 0))                                       
            Amount = float(request.form.get('Amount', 0))                                     
            ChannelId = float(request.form.get('ChannelID', 0))              
            SubscriptionId = float(request.form.get("SubscriptionId", 0))                    
            AccountId = float(request.form.get('AccountId', 0))
            ProviderId = float(request.form.get('ProviderId', 0))

            # Ensure scaler is available
            scaled_data = scaler.transform([[Outliers, PricingStrategy, ProductCategory_airtime, 
                                             ProductCategory_financial_services, Value, Amount, 
                                             ChannelId, SubscriptionId, AccountId, ProviderId]])

            # Predict with the loaded model
            prediction = model.predict(scaled_data)

            # Output the prediction (consider passing it to a template)
            print(prediction)
            return render_template('predicter.html', prediction=prediction)

        except Exception as e:
            # Print the error to the console and render an error page or message
            print(f"Error occurred: {e}")
            return render_template('predicter.html', error="An error occurred. Please check your inputs.")
    
    # Render form for GET requests
    return render_template('predicter.html')

if __name__=='__main__':
    app.run(debug=True)