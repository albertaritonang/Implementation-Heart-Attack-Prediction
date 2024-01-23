from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

# Load dataset
df_heart = pd.read_csv('heart.csv')

# Preprocessing (seperti yang sudah Anda lakukan sebelumnya)
# ...

# Define the StandardScaler globally
std_scaler = StandardScaler()

# Split the data for training and testing
fitur_scaled = std_scaler.fit_transform(df_heart.drop('output', axis=1))
X_train, X_test, y_train, y_test = train_test_split(fitur_scaled, df_heart['output'], test_size=0.2, random_state=42)

# Load model KNN
knn_model = KNeighborsClassifier(n_neighbors=35)
knn_model.fit(X_train, y_train)

# Calculate accuracy on the test set
y_pred = knn_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on the test set:", accuracy)  # Print the accuracy

@app.route('/')
def home():
    return render_template('predict.html', prediction=None, input_data=None, class_accuracy=None, prediction_graph=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil nilai input dari formulir HTML
        input_features = [float(request.form.get(f)) for f in df_heart.columns[:-1]]
        input_scaled = std_scaler.transform([input_features])

        # Lakukan prediksi
        prediction = knn_model.predict(input_scaled)

        # Buat DataFrame untuk menampilkan data input dan prediksi
        input_data = pd.DataFrame([input_features], columns=df_heart.columns[:-1])
        input_data['Prediction'] = prediction[0]

        # Calculate accuracy for each class on the specific input data
        class_accuracy = knn_model.predict_proba(input_scaled)[0]

        # Generate a bar chart for prediction probabilities
        prediction_probabilities = knn_model.predict_proba(input_scaled)[0]

        # Convert the prediction probabilities to risk levels
        risk_levels = ['Resiko Rendah', 'Resiko Tinggi']
        predicted_risk = risk_levels[prediction[0]]

        # Plotting the bar chart
        plt.bar(risk_levels, prediction_probabilities, color=['green', 'red'])
        plt.title('Prediction Probabilities')
        plt.xlabel('Resiko')
        plt.ylabel('Probabilitas')

        # Convert the plot to base64 format for embedding in HTML
        img_buf = BytesIO()
        plt.savefig(img_buf, format='png')
        img_buf.seek(0)
        img_str = "data:image/png;base64," + base64.b64encode(img_buf.read()).decode('utf-8')
        plt.close()

        return render_template('predict.html', prediction=predicted_risk, input_data=input_data.to_html(index=False, classes='table table-bordered table-hover'), class_accuracy=class_accuracy, prediction_graph=img_str)

    except Exception as e:
        print("Error during prediction:", str(e))
        return render_template('predict.html', prediction="Error during prediction", input_data=None, class_accuracy=None, prediction_graph=None)

if __name__ == '__main__':
    app.run(debug=True)
