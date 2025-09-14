import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

# Import your class and plotting functions from Stylometry.py
from Stylometry import PersianTextPreprocessor, plot_word_length_distribution, \
    plot_sentence_length_distribution, plot_vocabulary_richness, \
    plot_word_frequency_distribution, plot_ngram_frequencies, \
    plot_pos_distribution, plot_alphabet_distribution

# --- Configuration ---
# Update these paths to be relative to your app.py file
POS_TAGGER_MODEL_PATH = os.path.join("models", "pos_tagger.model")
LEXICONS_DIR_PATH = "lexicons"
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    """Renders the homepage with the file upload form."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_text():
    """Handles the file upload and analysis."""
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(request.url)

    if file:
        # Read the uploaded file content
        try:
            # Read file as utf-8 text
            persian_text = file.read().decode('utf-8').lstrip('\ufeff')
        except Exception as e:
            return f"Error reading file: {e}"

        # --- Initialize and Run Analysis ---
        preprocessor = PersianTextPreprocessor(
            pos_tagger_model=POS_TAGGER_MODEL_PATH,
            lexicons_dir=LEXICONS_DIR_PATH
        )
        analysis_results = preprocessor.comprehensive_text_analysis(persian_text)

        # --- Generate Plots ---
        plots = {}
        plots['word_length'] = plot_word_length_distribution(
            analysis_results['word_length']['word_length_distribution'],
            title="Mendenhall's Characteristic Curve"
        )
        plots['sentence_length'] = plot_sentence_length_distribution(
            analysis_results['sentence_length']['by_tokens']['sentence_length_distribution'],
            unit='tokens'
        )
        plots['vocab_richness'] = plot_vocabulary_richness(analysis_results['vocabulary_richness'])
        plots['word_freq'] = plot_word_frequency_distribution(analysis_results['word_frequencies']['top_100_words'][:20])
        plots['ngrams_unigrams'] = plot_ngram_frequencies(analysis_results['ngrams']['unigrams']['top_10'], "Unigrams")
        plots['ngrams_bigrams'] = plot_ngram_frequencies(analysis_results['ngrams']['bigrams']['top_10'], "Bigrams")
        plots['pos_dist'] = plot_pos_distribution(analysis_results['pos_analysis'])
        plots['alphabet_dist'] = plot_alphabet_distribution(analysis_results['alphabet_distribution'])


        # --- Render the Results Page ---
        # Pass both the analysis data and the plot images to the template
        return render_template('results.html', results=analysis_results, plots=plots)

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)