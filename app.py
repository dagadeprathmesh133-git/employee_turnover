import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'logistic_model.pkl')
model = None

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")

# HTML Template with modern, sleek styling
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employee Analytics Predictor</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
            --glass-bg: rgba(255, 255, 255, 0.05);
            --glass-border: rgba(255, 255, 255, 0.12);
            --accent-color: #6366f1;
            --accent-hover: #4f46e5;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background: var(--bg-gradient);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px 0;
        }

        .main-card {
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: 24px;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
            padding: 40px;
            max-width: 800px;
            width: 100%;
        }

        .form-label {
            font-weight: 600;
            font-size: 0.875rem;
            letter-spacing: 0.5px;
            color: var(--text-muted);
            margin-bottom: 8px;
        }

        .form-control, .form-select {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--glass-border);
            border-radius: 12px;
            color: var(--text-main);
            padding: 12px 16px;
            transition: all 0.3s ease;
        }

        .form-control:focus, .form-select:focus {
            background: rgba(15, 23, 42, 0.8);
            border-color: var(--accent-color);
            box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.25);
            color: var(--text-main);
        }

        .form-select option {
            background-color: #0f172a;
            color: #f8fafc;
        }

        .btn-predict {
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            border: none;
            border-radius: 12px;
            color: white;
            font-weight: 700;
            padding: 14px;
            font-size: 1rem;
            letter-spacing: 0.5px;
            transition: all 0.3s ease;
            box-shadow: 0 8px 20px rgba(99, 102, 241, 0.3);
        }

        .btn-predict:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 25px rgba(99, 102, 241, 0.5);
        }

        #result-box {
            display: none;
            margin-top: 30px;
            padding: 20px;
            border-radius: 16px;
            text-align: center;
            font-weight: 700;
            font-size: 1.2rem;
            animation: fadeIn 0.5s ease-in-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

<div class="container d-flex justify-content-center">
    <div class="main-card">
        <h2 class="text-center fw-bold mb-2">Employee Retention Predictor</h2>
        <p class="text-center text-secondary mb-4">Fill in the features below to make a prediction using your model.</p>
        
        <form id="predictionForm">
            <div class="row g-3">
                <div class="col-md-6">
                    <label class="form-label">Education Level</label>
                    <select class="form-select" name="Education" required>
                        <option value="0">Bachelors (0)</option>
                        <option value="1">Masters (1)</option>
                        <option value="2">PHD (2)</option>
                    </select>
                </div>
                <div class="col-md-6">
                    <label class="form-label">Joining Year</label>
                    <input type="number" class="form-control" name="JoiningYear" value="2018" required>
                </div>
                <div class="col-md-6">
                    <label class="form-label">City Code</label>
                    <select class="form-select" name="City" required>
                        <option value="0">Bangalore (0)</option>
                        <option value="1">Pune (1)</option>
                        <option value="2">New Delhi (2)</option>
                    </select>
                </div>
                <div class="col-md-6">
                    <label class="form-label">Payment Tier</label>
                    <select class="form-select" name="PaymentTier" required>
                        <option value="1">Tier 1</option>
                        <option value="2">Tier 2</option>
                        <option value="3">Tier 3</option>
                    </select>
                </div>
                <div class="col-md-6">
                    <label class="form-label">Age</label>
                    <input type="number" class="form-control" name="Age" value="28" required>
                </div>
                <div class="col-md-6">
                    <label class="form-label">Gender</label>
                    <select class="form-select" name="Gender" required>
                        <option value="0">Male (0)</option>
                        <option value="1">Female (1)</option>
                    </select>
                </div>
                <div class="col-md-6">
                    <label class="form-label">Ever Benched</label>
                    <select class="form-select" name="EverBenched" required>
                        <option value="0">No (0)</option>
                        <option value="1">Yes (1)</option>
                    </select>
                </div>
                <div class="col-md-6">
                    <label class="form-label">Experience In Current Domain</label>
                    <input type="number" class="form-control" name="ExperienceInCurrentDomain" value="3" required>
                </div>
            </div>

            <button type="submit" class="btn btn-predict w-100 mt-4">Predict Outcome</button>
        </form>

        <div id="result-box"></div>
    </div>
</div>

<script>
document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    const formData = new FormData(this);
    const resultBox = document.getElementById('result-box');
    
    resultBox.style.display = 'block';
    resultBox.className = 'alert alert-info';
    resultBox.innerText = 'Calculating prediction...';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        
        if (data.error) {
            resultBox.className = 'alert alert-danger';
            resultBox.innerText = 'Error: ' + data.error;
        } else {
            resultBox.className = 'alert alert-success';
            resultBox.innerText = 'Prediction Class Result: ' + data.prediction;
        }
    } catch (err) {
        resultBox.className = 'alert alert-danger';
        resultBox.innerText = 'Error processing request.';
    }
});
</script>

</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model failed to load on server.'}), 500

    try:
        # Features ordered exactly as expected by your pickled model
        feature_keys = [
            'Education', 'JoiningYear', 'City', 'PaymentTier',
            'Age', 'Gender', 'EverBenched', 'ExperienceInCurrentDomain'
        ]
        
        features = [float(request.form.get(key, 0)) for key in feature_keys]
        final_features = [np.array(features)]
        
        prediction = model.predict(final_features)
        
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
