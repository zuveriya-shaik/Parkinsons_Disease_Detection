import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback

# Load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv('parkinsons_1.csv')
    X = data.drop(['name', 'status'], axis=1)
    y = data['status']
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    return X_train, X_test, y_train, y_test

# Function to build models
def build_model(model_type, input_shape):
    model = Sequential()
    if model_type == 'LSTM':
        model.add(LSTM(units=100, activation='relu', return_sequences=True, input_shape=input_shape))
        model.add(LSTM(units=100, activation='relu'))
        model.add(Dropout(0.1))
    elif model_type == 'GRU':
        model.add(GRU(units=100, activation='relu', return_sequences=True, input_shape=input_shape))
        model.add(GRU(units=100, activation='relu'))
        model.add(Dropout(0.1))
    elif model_type == 'Hybrid LSTM-GRU':
        model.add(LSTM(units=100, activation='relu', return_sequences=True, input_shape=input_shape))
        model.add(LSTM(units=100, activation='relu', return_sequences=True))
        model.add(Dropout(0.1))
        model.add(GRU(units=256, return_sequences=True))
        model.add(Dense(units=128, activation='relu'))
        model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    return model

# Train and evaluate the model
def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    class StreamlitCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            st.write(f"Epoch {epoch + 1}/50 - Loss: {logs['loss']:.4f}, Accuracy: {logs['accuracy']:.4f}")
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[StreamlitCallback()], verbose=0)

    y_pred = (model.predict(X_test) > 0.5).astype(int).reshape(-1)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    return acc, prec, rec, f1, cm, report

# Initialize session state
if 'model_results' not in st.session_state:
    st.session_state.model_results = {'LSTM': None, 'GRU': None, 'Hybrid LSTM-GRU': None}

# Load data
X_train, X_test, y_train, y_test = load_data()
feature_size = X_train.shape[2]

# Streamlit interface
st.title("Parkinson's Disease Detection")
st.write("Train models to detect Parkinson's disease from voice data.")

# Sidebar buttons for training models
model_type = st.sidebar.selectbox("Select Model to Train", ['LSTM', 'GRU', 'Hybrid LSTM-GRU'])
if st.sidebar.button("Train Model"):
    st.write(f"Training {model_type} Model...")
    model = build_model(model_type, (None, feature_size))
    acc, prec, rec, f1, cm, report = train_and_evaluate(model, X_train, y_train, X_test, y_test)
    st.session_state.model_results[model_type] = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'Confusion Matrix': cm,
        'Report': report
    }
    st.success(f"{model_type} Model Trained Successfully!")

# Sidebar button for comparing models
if st.sidebar.button("Compare Models"):
    untrained_models = [model for model, results in st.session_state.model_results.items() if results is None]
    if untrained_models:
        st.error(f"Please train all models first! Remaining models to train: {untrained_models}")
    else:
        st.write("Comparison of Model Performance:")
        comparison_df = pd.DataFrame({
            'Model': ['LSTM', 'GRU', 'Hybrid LSTM-GRU'],
            'Accuracy': [st.session_state.model_results['LSTM']['Accuracy'], st.session_state.model_results['GRU']['Accuracy'], st.session_state.model_results['Hybrid LSTM-GRU']['Accuracy']],
            'Precision': [st.session_state.model_results['LSTM']['Precision'], st.session_state.model_results['GRU']['Precision'], st.session_state.model_results['Hybrid LSTM-GRU']['Precision']],
            'Recall': [st.session_state.model_results['LSTM']['Recall'], st.session_state.model_results['GRU']['Recall'], st.session_state.model_results['Hybrid LSTM-GRU']['Recall']],
            'F1-Score': [st.session_state.model_results['LSTM']['F1-Score'], st.session_state.model_results['GRU']['F1-Score'], st.session_state.model_results['Hybrid LSTM-GRU']['F1-Score']],
        }).set_index('Model')
        st.dataframe(comparison_df)

        # Plot comparison
        st.write("### Performance Comparison")
        fig, ax = plt.subplots(figsize=(10, 6))
        comparison_df.plot(kind='bar', alpha=0.8, ax=ax)
        plt.title("Model Performance Metrics")
        plt.ylabel("Scores")
        plt.xlabel("Models")
        plt.xticks(rotation=0)
        st.pyplot(fig)
