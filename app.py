from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load your CSV file
df = pd.read_csv('bus_crowd_data_holidays_OG_copy(changed)_1.csv')
# df = pd.read_csv('bus_crowd_data_holidays_OG.csv')


# Assuming 'crowded' is your target variable, and other columns are features
# Include 'holidays' in features
X = df[['time_of_day', 'weather_condition', 'holidays']]
y = df['crowded']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create a Logistic Regression model
model = LogisticRegression()

# Fit the model with training data
model.fit(X_train, y_train)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        time_of_day = int(request.form['time_of_day'])
        weather_condition = int(request.form['weather_condition'])
        # Get the 'holidays' value from the form
        holidays = int(request.form['holidays'])

        # Create a new data point with the user input
        # Include 'holidays'
        new_data = [[time_of_day, weather_condition, holidays]]

        # Make a prediction on the new data
        prediction = model.predict(new_data)

        # Return the prediction result
        result = "The bus is crowded." if prediction[0] else "The bus is not crowded."
        return result


if __name__ == '__main__':
    app.run(debug=True)
