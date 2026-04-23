# 🎓 Student Performance Predictor

A machine learning application to predict student final grades (G3) based on various academic and personal factors. Built with FastAPI for the backend, Streamlit for the frontend, and scikit-learn for modeling.

## 📊 Dataset

The application uses the [Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance) from UCI Machine Learning Repository. It contains data about students from two Portuguese schools, including grades, demographic, social, and school-related features.

**Key Features Used for Prediction:**
- `studytime`: Weekly study time (1-4)
- `failures`: Number of past class failures (0-3)
- `absences`: Number of school absences (0-93)
- `Medu`: Mother's education level (0-4)
- `Fedu`: Father's education level (0-4)
- `famrel`: Quality of family relationships (1-5)
- `goout`: Going out with friends (1-5)
- `Dalc`: Workday alcohol consumption (1-5)
- `Walc`: Weekend alcohol consumption (1-5)
- `health`: Current health status (1-5)

## 🚀 Features

- **Grade Prediction**: Predict final exam scores (0-20) using regression models
- **Model Comparison Dashboard**: Compare performance of different ML models (MAE, R² scores)
- **Feature Importance**: Visualize which features contribute most to predictions
- **Interactive UI**: User-friendly Streamlit interface with sliders and tabs

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/student-performance-ml.git
   cd student-performance-ml
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model:**
   ```bash
   cd backend
   python train.py
   ```

## 🏃‍♂️ Usage

1. **Start the backend API:**
   ```bash
   cd backend
   python -m uvicorn main:app --reload
   ```
   The API will be available at `http://127.0.0.1:8000`

2. **Start the frontend:**
   ```bash
   cd frontend
   streamlit run app.py
   ```
   The web app will open at `http://localhost:8501`

3. **Use the application:**
   - **Predict Final Grade Tab**: Adjust the sliders for student features and click "Predict" to get the estimated final score.
   - **Model Comparison Tab**: View performance metrics and feature importances for different models.

## 📁 Project Structure

```
student-performance-ml/
├── backend/
│   ├── main.py              # FastAPI server
│   ├── train.py             # Model training script
│   ├── model.pkl            # Trained model
│   ├── scaler.pkl           # Feature scaler
│   ├── columns.pkl          # Feature column names
│   ├── metrics.pkl          # Model performance metrics
│   └── feature_importances.pkl  # Feature importances
├── frontend/
│   └── app.py               # Streamlit web app
├── dataset/
│   └── student_data.csv     # Dataset
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── docker-compose.yml       # Docker configuration
```

## 🤖 Models Used

- **Linear Regression**
- **Random Forest Regressor** (Best performing)
- **Decision Tree Regressor**
- **K-Nearest Neighbors Regressor**

**Best Model Performance:**
- Mean Absolute Error (MAE): ~3.49
- R² Score: ~0.13

## 🔧 API Endpoints

- `GET /`: Health check
- `POST /predict`: Predict final grade
  - **Input:** JSON with feature values
  - **Output:** `{"prediction": score}` (0-20)

Example request:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "studytime": 2,
       "failures": 0,
       "absences": 5,
       "Medu": 2,
       "Fedu": 2,
       "famrel": 4,
       "goout": 3,
       "Dalc": 1,
       "Walc": 1,
       "health": 3
     }'
```

## 📈 Model Evaluation

The models are evaluated using:
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual grades
- **R² Score**: Proportion of variance in the dependent variable explained by the model

Lower MAE and higher R² indicate better performance.

## 🐳 Docker Support

To run with Docker:

```bash
docker-compose up --build
```

## 📝 License

This project is for educational purposes. The dataset is publicly available from UCI ML Repository.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Contact

For questions or suggestions, please open an issue on GitHub.