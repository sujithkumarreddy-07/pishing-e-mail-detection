# Phishing Email Detection Model

## Objective
Build a machine learning model using Scikit-learn to classify emails as Phishing or Safe.

## Features
- Train on phishing and legitimate email data
- Extract textual features using TF-IDF
- Analyze URLs and suspicious keywords
- Classify emails as Phishing or Safe
- Display accuracy and confusion matrix
- Test new email messages

## Files
- `phishing_email_detection.py` - main ML program
- `emails.csv` - sample training dataset
- `requirements.txt` - Python dependencies
- `README.md` - project instructions
- `confusion_matrix.png` - generated after running the program

## Run
1. Install Python 3.
2. Open Command Prompt in this folder.
3. Run:
   `pip install -r requirements.txt`
4. Run:
   `python phishing_email_detection.py`

The program generates `confusion_matrix.png` and prints the accuracy, classification report, and predictions.
