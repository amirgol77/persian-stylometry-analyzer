# Persian Stylometry Analyzer

A web-based tool built with Flask and Hazm to perform comprehensive stylistic analysis of Persian texts. Users can upload a `.txt` file and get detailed reports on vocabulary richness, sentence complexity, POS tag distribution, and more.

## Features
- Basic statistics (character and word counts)
- Vocabulary richness analysis (TTR, Hapax Legomena, etc.)
- Word and sentence length distribution plots
- N-gram (unigram, bigram, trigram) frequency analysis
- Part-of-Speech (POS) tagging and distribution
- SFG stylistic features analysis
- Alphabet and punctuation distribution

## Setup and Installation

**1. Clone the repository:**
```bash
git clone https://github.com/YourUsername/persian-stylometry-analyzer.git
cd persian-stylometry-analyzer
```
2. Create and activate a virtual environment:
```bash
python -m venv venv
```
# On Windows
```bash
venv\Scripts\activate
```
# On macOS/Linux
```bash
source venv/bin/activate
```
3. Install the required packages:
```bash
pip install -r requirements.txt
```
How to Run
With your virtual environment active, run the Flask application:
```bash
python -m flask run
```
Then, open your web browser and navigate to http://127.0.0.1:5000.
