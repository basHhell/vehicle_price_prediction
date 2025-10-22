from flask import Flask,render_template,request,redirect,url_for
from catboost import CatBoostRegressor
import pandas as pd

app=Flask(__name__)

# Load the model
model=CatBoostRegressor()
model.load_model("vehicle_model.cbm")

model_features = [
    'name', 'make', 'description', 'model', 'year', 'engine', 'cylinders',
    'fuel', 'mileage', 'transmission', 'trim', 'body', 'doors',
    'exterior_color', 'interior_color', 'drivetrain'
]

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    try:
        input_data = {
            'name': request.form.get('name', 'Unknown'),
            'make': request.form.get('make', 'Unknown'),
            'description': request.form.get('description', 'No description'),
            'model': request.form.get('model', 'Unknown'),
            'year': int(request.form.get('year') or 2020),
            'engine': request.form.get('engine', 'Unknown'),
            'cylinders': float(request.form.get('cylinders') or 4),
            'fuel': request.form.get('fuel', 'Gasoline'),
            'mileage': float(request.form.get('mileage') or 0),
            'transmission': request.form.get('transmission', 'Automatic'),
            'trim': request.form.get('trim', 'Base'),
            'body': request.form.get('body', 'Sedan'),
            'doors': int(request.form.get('doors') or 4),  # Default to 4 doors
            'exterior_color': request.form.get('exterior_color', 'White'),
            'interior_color': request.form.get('interior_color', 'Black'),
            'drivetrain': request.form.get('drivetrain', 'FWD')
        }



        # Create DataFrame for prediction
        df = pd.DataFrame([input_data], columns=model_features)

        # Handling the null values
        for col in model_features:
            df[col]=df[col].fillna("unknown")

        # Predict
        predicted_price=model.predict(df)[0]

        return render_template('results.html', price=round(predicted_price, 2))

    except Exception as e:
        return f"Error: {e}"


if __name__ == '__main__':
    app.run(debug=True)
