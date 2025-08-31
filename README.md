📰 AI News Verifier
An intelligent web application built with Python and Streamlit that uses a fine-tuned BERT model to detect and classify fake news articles. This tool provides a simple interface to paste news text and get an instant, AI-powered analysis of its likely authenticity.

 <!-- Replace with your live Streamlit URL after deployment -->

✨ Key Features
State-of-the-Art AI: Utilizes a BERT (Bidirectional Encoder Representations from Transformers) model, fine-tuned on a diverse news dataset for high accuracy.

Simple Web Interface: An elegant and user-friendly UI built with Streamlit where users can paste article text and get instant results.

Confidence Score: Provides not just a "REAL" or "FAKE" classification, but also a confidence score to indicate how certain the model is about its prediction.

End-to-End Project: Covers the entire machine learning lifecycle, from data preprocessing and model training to deployment as a live web application.

Handles Large Files: Correctly configured with Git LFS to manage the large model files required for deployment.

🛠️ Technology Stack
Backend & Modeling:

Python 3.10+

PyTorch: The deep learning framework used for model training.

Hugging Face transformers: For accessing the pre-trained BERT model and tokenizer.

Scikit-learn: For splitting the dataset into training and validation sets.

Pandas: For loading, cleaning, and managing the datasets.

Frontend & Deployment:

Streamlit: For creating and deploying the interactive web application.

Git & Git LFS: For version control and handling large model files.

🚀 Getting Started
Follow these instructions to set up and run the project on your local machine.

Prerequisites
Python 3.10 or higher

Git and Git LFS installed on your system. You can download Git LFS from here.

1. Clone the Repository
First, clone the project to your local machine.

git clone [https://github.com/YourUsername/your-repo-name.git](https://github.com/YourUsername/your-repo-name.git)
cd your-repo-name

2. Set Up Git LFS
Initialize Git LFS to pull the large model files correctly.

git lfs install
git lfs pull

3. Install Dependencies
Install all the necessary Python libraries using the requirements.txt file.

pip install -r requirements.txt

4. Place the Datasets
This project requires three data files. Download them and place them in the root of your project folder:

fake.csv

true.csv

articles1.csv (From the "All the News" Kaggle dataset)

🏃‍♂️ How to Run the Application
The project is a two-step process: first, you train the model, and then you run the web app.

Step 1: Train the AI Model
Run the training script from your terminal. This will process the datasets and create a bert_fake_news_model folder containing your fine-tuned model.

python train_model.py

Note: This process can take a significant amount of time, depending on your computer's hardware (CPU vs. GPU).

Step 2: Launch the Streamlit Web App
Once the model has been trained and saved, you can launch the web application.

streamlit run app.py

Your default web browser will automatically open with the AI News Verifier running and ready to use!

🌐 Deployment
This application is deployed on Streamlit Community Cloud. The deployment process is configured to:

Install dependencies from requirements.txt.

Use Git LFS to download the trained model files.

Launch the app.py script.

⚠️ Disclaimer
This AI tool is designed for educational and demonstration purposes. Its accuracy is limited by the dataset it was trained on and it should not be used as a substitute for professional fact-checking. Always verify information from multiple reputable sources.
