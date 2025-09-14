import re
import string
import os
import math
import hazm
import numpy as np
from nltk.util import ngrams
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from bidi.algorithm import get_display
import arabic_reshaper
import itertools
import pandas as pd
import json
from pathlib import Path
import io
import base64

class PersianTextPreprocessor:

    def __init__(self, pos_tagger_model, lexicons_dir):

        self.normalizer = hazm.Normalizer()
        self.lemmatizer = hazm.Lemmatizer()
        self.sent_tokenizer = hazm.SentenceTokenizer()
        self.word_tokenizer = hazm.WordTokenizer()
        self.pos_tagger = hazm.POSTagger(model=pos_tagger_model)

        self.persian_punctuations = """\.:!،؛«»\-()؟…"'\[\]{}"""

        self.pos_tag_descriptions = {
            'NOUN': 'Noun', 'NOUN,EZ': 'Noun with Ezafe', 'ADJ': 'Adjective',
            'ADJ,EZ': 'Adjective with Ezafe', 'ADV': 'Adverb', 'VERB': 'Verb',
            'AUX': 'Auxiliary Verb', 'PRON': 'Pronoun', 'DET': 'Determiner',
            'PREP': 'Preposition', 'POSTP': 'Postposition', 'NUM': 'Number',
            'CONJ': 'Conjunction', 'PUNC': 'Punctuation', 'INTJ': 'Interjection',
            'RES': 'Residual', 'CL': 'Classifier', 'NEG': 'Negation',
            'DEF': 'Definite marker', 'PART': 'Particle',
        }

        # SFG Lexicons Initialization
        lex_path = Path(lexicons_dir) # Use the provided path
        self.lexicons = {}
        for name in ("conjunction", "modality", "comment", "appraisal"):
            file_path = lex_path / f"{name}.json"
            if file_path.exists():
                self.lexicons[name] = json.loads(file_path.read_text(encoding="utf8"))
            else:
                print(f"Warning: Lexicon file not found at {file_path}")
                self.lexicons[name] = {} # Initialize as empty dict if not found


    # --- Text Cleaning and Basic Processing Methods ---
    def normalize_text(self, text):
        return self.normalizer.normalize(text)

    def remove_punctuation(self, text):
        translator = str.maketrans('', '', string.punctuation + self.persian_punctuations)
        return text.translate(translator)

    def remove_numbers(self, text):
        persian_digits = "۰۱۲۳۴۵۶۷۸۹"
        english_digits = "0123456789"
        translator = str.maketrans('', '', persian_digits + english_digits)
        return text.translate(translator)

    def remove_whitespace(self, text):
        return re.sub(r'\s+', ' ', text).strip()

    def tokenize(self, text):
        return self.word_tokenizer.tokenize(text)

    def sentence_tokenize(self, text):
        return self.sent_tokenizer.tokenize(text)

    def lemmatize_words(self, text):
        words = self.tokenize(text)
        return ' '.join([self.lemmatizer.lemmatize(word) for word in words])


    def clean_text(self, text, remove_puncs=True, remove_nums=True, normalize=True, lemmatize=True):
        if normalize:
            text = self.normalize_text(text)
        if remove_puncs:
            text = self.remove_punctuation(text)
        if remove_nums:
            text = self.remove_numbers(text)
        text = self.remove_whitespace(text)
        if lemmatize:
            text = self.lemmatize_words(text)
        return text


    def calculate_vocab_richness(self, tokens):
        total_tokens = len(tokens)
        total_unique_tokens = len(set(tokens))

        token_freq = Counter(tokens)

        # hapax legomena
        hapax_count = sum(1 for word, freq in token_freq.items() if freq == 1)
        hapax_legomena_words = [word for word, freq in token_freq.items() if freq == 1]

        # dis legomena
        dis_count = sum(1 for word, freq in token_freq.items() if freq == 2)
        dis_legomena_words = [word for word, freq in token_freq.items() if freq == 2]

        metrics = {
            # Type-Token Ratio (TTR) - basic lexical diversity
            'ttr': total_unique_tokens / total_tokens if total_tokens > 0 else 0,

            # Root TTR - reduces the impact of text length
            'root_ttr': total_unique_tokens / math.sqrt(total_tokens) if total_tokens > 0 else 0,

            # Log TTR - another way to reduce text length impact
            'log_ttr': total_unique_tokens / math.log10(total_tokens) if total_tokens > 1 else 0,

            # Hapax Legomena Ratio - proportion of words occurring once
            'hapax_ratio': hapax_count / total_tokens if total_tokens > 0 else 0,

            # Hapax Legomena to Total Unique Ratio
            'hapax_unique_ratio': hapax_count / total_unique_tokens if total_unique_tokens > 0 else 0,

            # Yule's K - measure of vocabulary richness based on repetition
            'yule_k': 10000 * (sum((freq / total_tokens) ** 2 for freq in token_freq.values()) - (1 / total_tokens)) if total_tokens > 0 else 0,

            # Counts for reference
            'total_tokens': total_tokens,
            'unique_tokens': total_unique_tokens,
            'hapax_count': hapax_count,
            'dis_count': dis_count,
            'hapax_legomena_words': hapax_legomena_words,
            'dis_legomena_words': dis_legomena_words
        }

        return metrics

    def get_most_frequent_words(self, tokens, top_n=100):
        return Counter(tokens).most_common(top_n)

    def get_freq_distribution(self, tokens):
        return dict(Counter(tokens))

    def analyze_word_lengths(self, tokens):
        if not tokens: return {}
        word_lengths = [len(word) for word in tokens]
        return {
            'mean_word_length': np.mean(word_lengths),
            'median_word_length': np.median(word_lengths),
            'min_word_length': min(word_lengths),
            'max_word_length': max(word_lengths),
            'std_word_length': np.std(word_lengths),
            'word_length_distribution': Counter(word_lengths)
        }

    def analyze_sentence_lengths(self, text, by_tokens=True):
        sentences = self.sentence_tokenize(text)
        if not sentences: return {}
        if by_tokens:
            sentence_lengths = [len(self.tokenize(sent)) for sent in sentences]
            unit = "tokens"
        else:
            sentence_lengths = [len(sent) for sent in sentences]
            unit = "characters"

        return {
            'sentence_count': len(sentences),
            f'mean_sentence_length_{unit}': np.mean(sentence_lengths),
            f'median_sentence_length_{unit}': np.median(sentence_lengths),
            f'min_sentence_length_{unit}': min(sentence_lengths),
            f'max_sentence_length_{unit}': max(sentence_lengths),
            f'std_sentence_length_{unit}': np.std(sentence_lengths),
            'sentence_length_distribution': Counter(sentence_lengths)
        }

    # --- N-gram Analysis Methods ---
    def generate_ngrams(self, tokens, n=2):
        return list(ngrams(tokens, n))

    def get_ngram_frequencies(self, tokens, n=2, top_n=20):
        n_grams = self.generate_ngrams(tokens, n)
        return Counter(n_grams).most_common(top_n)

    def analyze_all_ngrams(self, tokens, max_n=3):
        results = {}
        for n in range(1, max_n + 1):
            ngram_name = {1: "unigrams", 2: "bigrams", 3: "trigrams"}.get(n, f"{n}-grams")
            ngram_list = self.generate_ngrams(tokens, n)
            if not ngram_list: continue

            ngram_freq = Counter(ngram_list)
            top_ngrams = ngram_freq.most_common(10)

            formatted_ngrams = []
            for gram, count in top_ngrams:
                if n == 1:
                    gram_text = gram[0]
                else:
                    gram_text = " ".join(gram)
                formatted_ngrams.append((gram_text, count))

            results[ngram_name] = {
                'total_unique': len(ngram_freq),
                'total_count': len(ngram_list),
                'top_10': formatted_ngrams
            }

        return results

    # --- POS Tagging Methods ---
    def get_pos_tags(self, text, normalized=True):
        if normalized:
            text = self.normalize_text(text)
        tokens = self.tokenize(text)
        return self.pos_tagger.tag(tokens)


    def get_pos_tag_description(self, tag):
        return self.pos_tag_descriptions.get(tag, tag)

    def analyze_pos_tags(self, text, normalized=True):

        # Get POS tags
        # Explicitly pass normalized as a keyword argument
        tagged_tokens = self.get_pos_tags(text, normalized=normalized)

        if not tagged_tokens:
            return {"error": "No tagged tokens found"}

        # Extract tags
        tags = [tag for _, tag in tagged_tokens]

        # Count frequencies
        tag_freq = Counter(tags)

        # Calculate percentages
        total_tokens = len(tags)
        tag_percentages = {tag: count/total_tokens*100 for tag, count in tag_freq.items()}

        # Prepare results with descriptions
        results = {
            'total_tokens': total_tokens,
            'unique_tags': len(tag_freq),
            'tag_counts': dict(tag_freq.most_common()),
            'tag_percentages': tag_percentages,
            'tag_descriptions': {tag: self.get_pos_tag_description(tag) for tag in tag_freq.keys()},
            'most_common': [(tag, count, self.get_pos_tag_description(tag))
                            for tag, count in tag_freq.most_common(10)]
        }

        return results

    def get_pos_tag_examples(self, text, top_n=3):

        # Get POS tags
        tagged_tokens = self.get_pos_tags(text)

        # Collect examples for each tag
        tag_examples = {}

        for word, tag in tagged_tokens:
            if tag not in tag_examples:
                tag_examples[tag] = []

            if word not in tag_examples[tag]:
                tag_examples[tag].append(word)

            # Stop once we have enough examples
            if len(tag_examples[tag]) >= top_n:
                continue

        # Add descriptions
        for tag in tag_examples:
            tag_examples[tag] = {
                'description': self.get_pos_tag_description(tag),
                'examples': tag_examples[tag]
            }

        return tag_examples

    # --- Sentence Complexity Method
    def analyze_sentence_complexity(self, text):
        normalized_text = self.normalize_text(text)
        sentences = self.sent_tokenizer.tokenize(normalized_text)
        verb_categories = defaultdict(int)
        total_sentences = 0

        for sentence in sentences:
            if not sentence.strip(): continue
            total_sentences += 1
            words = self.word_tokenizer.tokenize(sentence)
            pos_tags = self.pos_tagger.tag(words)
            verb_count = sum(1 for _, pos in pos_tags if pos.startswith('V'))
            verb_categories[verb_count] += 1

        if total_sentences == 0: return {}

        simple = verb_categories.get(1, 0)
        complex_s = sum(c for vc, c in verb_categories.items() if vc > 1)
        no_verb = verb_categories.get(0, 0)

        return {
            'total_sentences': total_sentences,
            'verb_distribution': dict(sorted(verb_categories.items())),
            'simple_sentences': simple,
            'complex_sentences': complex_s,
            'no_verb_sentences': no_verb,
            'simple_percentage': (simple / total_sentences) * 100,
            'complex_percentage': (complex_s / total_sentences) * 100,
            'no_verb_percentage': (no_verb / total_sentences) * 100,
        }

    # --- SFG Stylistic Features Method
    def analyze_sfg_features(self, text):
        sentences = self.sent_tokenizer.tokenize(self.normalize_text(text))
        tagged_sents = [self.pos_tagger.tag(self.word_tokenizer.tokenize(s)) for s in sentences]

        counts = Counter()
        for sent in tagged_sents:
            for word, _ in sent:
                w = word.lower()
                for lex in self.lexicons.values():
                    if w in lex:
                        for k, v in lex[w].items():
                            counts[(k, v)] += 1

        total = sum(counts.values()) or 1
        return {
            'total_features_found': sum(counts.values()),
            'feature_counts': dict(counts),
            'feature_relative_frequencies': {f"{attr}|{val}": (freq / total) for (attr, val), freq in counts.items()}
        }

    # --- Alphabet Distribution Analysis Method ---
    def analyze_alphabet_distribution(self, text):
        """
        Calculates the frequency and percentage of each Persian alphabet character in the text.
        """
        # A string containing all Persian alphabet characters
        persian_alphabets = "ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی"

        # Normalize the text for consistent character representation (e.g., ي vs ی)
        normalized_text = self.normalizer.normalize(text)

        # Count only the characters that are in the Persian alphabet
        alphabet_counts = Counter([char for char in normalized_text if char in persian_alphabets])

        total_alphabets = sum(alphabet_counts.values())

        if total_alphabets == 0:
            return {'counts': {}, 'percentages': {}, 'total_alphabets': 0}

        # Calculate the percentage for each alphabet
        alphabet_percentages = {char: (count / total_alphabets) * 100 for char, count in alphabet_counts.items()}

        # Sort the results alphabetically for consistent output
        sorted_counts = dict(sorted(alphabet_counts.items()))
        sorted_percentages = dict(sorted(alphabet_percentages.items()))

        return {
            'counts': sorted_counts,
            'percentages': sorted_percentages,
            'total_alphabets': total_alphabets
        }


    def analyze_punctuation_distribution(self, text):
        """
        Calculates the frequency and percentage of each punctuation mark in the text.
        """
        # A string containing common English and Persian punctuation marks
        punctuations = string.punctuation + self.persian_punctuations

        # Count only the characters that are in the punctuations string
        punctuation_counts = Counter([char for char in text if char in punctuations])

        total_punctuations = sum(punctuation_counts.values())

        if total_punctuations == 0:
            return {'counts': {}, 'percentages': {}, 'total_punctuations': 0}

        # Calculate the percentage for each punctuation mark
        punctuation_percentages = {char: (count / total_punctuations) * 100 for char, count in punctuation_counts.items()}

        # Sort the results by count in descending order for consistent output
        sorted_counts = dict(sorted(punctuation_counts.items(), key=lambda item: item[1], reverse=True))
        sorted_percentages = {char: (sorted_counts[char] / total_punctuations) * 100 for char in sorted_counts}


        return {
            'counts': sorted_counts,
            'percentages': sorted_percentages,
            'total_punctuations': total_punctuations
        }


    def comprehensive_text_analysis(self, text, preprocess=True, **preprocess_args):

        # Default preprocessing parameters if not specified
        default_args = {
            'remove_nums': True,
            'remove_puncs': True,
            'normalize': True,
            'lemmatize': True
        }

        # Update default args with any provided args
        preprocess_kwargs = {**default_args, **preprocess_args}

        # Initialize results
        analysis = {}

        # Step 1: Get both raw and processed text
        if preprocess:
            processed_text = self.clean_text(text, **preprocess_kwargs)
            processed_tokens = self.tokenize(processed_text)
            analysis['processed_text'] = processed_text
        else:
            processed_text = text
            processed_tokens = self.tokenize(text)

        raw_tokens = self.tokenize(text)

        # Step 2: Basic text statistics
        analysis['basic_stats'] = {
            'raw_char_count': len(text),
            'raw_token_count': len(raw_tokens),
            'processed_char_count': len(processed_text),
            'processed_token_count': len(processed_tokens)
        }

        # Step 3: Word length analysis
        analysis['word_length'] = self.analyze_word_lengths(processed_tokens)

        # Step 4: Sentence length analysis (both by tokens and characters)
        analysis['sentence_length'] = {
            'by_tokens': self.analyze_sentence_lengths(text, by_tokens=True),
            'by_chars': self.analyze_sentence_lengths(text, by_tokens=False)
        }

        # Step 5: Vocabulary richness
        analysis['vocabulary_richness'] = self.calculate_vocab_richness(processed_tokens)

        # Step 6: Word frequencies
        freq_dist = self.get_freq_distribution(processed_tokens)
        top_words = self.get_most_frequent_words(processed_tokens, top_n=100)

        analysis['word_frequencies'] = {
            'total_unique_words': len(freq_dist),
            'top_100_words': top_words
        }

        # Step 7: N-gram analysis
        analysis['ngrams'] = self.analyze_all_ngrams(processed_tokens, max_n=3)

        # Step 8: POS tagging analysis (new)
        analysis['pos_analysis'] = self.analyze_pos_tags(text)

        # Step 9: Sentence complexity analysis
        analysis['sentence_complexity'] = self.analyze_sentence_complexity(text)

        # Step 10: SFG features analysis
        analysis['sfg_features'] = self.analyze_sfg_features(text)

        # Step 11: Alphabet Distribution Analysis
        analysis['alphabet_distribution'] = self.analyze_alphabet_distribution(text)

        # Step 12: Punctuation Distribution Analysis
        analysis['punctuation_distribution'] = self.analyze_punctuation_distribution(text)


        return analysis
    
def plot_word_length_distribution(word_lengths, title="Word Length Distribution"):
    lengths = list(word_lengths.keys())
    counts = list(word_lengths.values())

    # Sort by length to ensure the plot is ordered correctly
    sorted_data = sorted(zip(lengths, counts))
    lengths, counts = zip(*sorted_data)

    plt.figure(figsize=(10, 6))
    plt.plot(lengths, counts, color='skyblue', marker='o', linestyle='-')
    plt.xlabel('Word Length (characters)')
    plt.ylabel('Frequency (Counts)')
    plt.title(title)
    plt.grid(True)
    plt.xticks(lengths)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close() # Close the plot to free up memory
    return plot_url


def plot_sentence_length_distribution(sentence_lengths, unit='tokens'):
    lengths = list(sentence_lengths.keys())
    counts = list(sentence_lengths.values())

    plt.figure(figsize=(8, 5))
    plt.bar(lengths, counts, color='lightgreen')
    plt.xlabel(f'Sentence Length ({unit})')
    plt.ylabel('Frequency')
    plt.title(f'Sentence Length Distribution ({unit})')
    plt.grid(True, axis='y')
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close() # Close the plot to free up memory
    return plot_url

def plot_vocabulary_richness(vocab_metrics, title="Vocabulary Richness Metrics"):
    metrics_to_plot = [
        'ttr', 'root_ttr', 'log_ttr', 'hapax_ratio',
        'hapax_unique_ratio'
    ]

    labels = [metric.replace('_', ' ').title() for metric in metrics_to_plot]
    values = [vocab_metrics[metric] for metric in metrics_to_plot]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, values, color='mediumpurple')
    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.ylabel("Value")
    plt.grid(True, axis='y')

    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close() # Close the plot to free up memory
    return plot_url

def reshape_persian(texts):
    return [get_display(arabic_reshaper.reshape(str(t))) for t in texts]

def plot_word_frequency_distribution(top_words, title="Top Word Frequencies (Persian)"):
    words, freqs = zip(*top_words)
    reshaped_words = reshape_persian(words)

    ranks = list(range(1, len(top_words) + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(ranks, freqs, color='coral', marker='o', linestyle='-')
    plt.xticks(ranks, reshaped_words, rotation=45, ha='right')
    plt.xlabel("Word Rank")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close() # Close the plot to free up memory
    return plot_url

def plot_ngram_frequencies(ngram_data, ngram_type="Unigrams", title_prefix="Top"):
    ngrams, freqs = zip(*ngram_data)
    reshaped_ngrams = reshape_persian(ngrams)

    ranks = list(range(1, len(ngram_data) + 1))

    plt.figure(figsize=(10, 5))
    plt.plot(ranks, freqs, color='steelblue', marker='o', linestyle='-')
    plt.xticks(ranks, reshaped_ngrams, rotation=45, ha='right')
    plt.xlabel(f"{ngram_type} Rank")
    plt.ylabel("Frequency")
    plt.title(f"{title_prefix} {ngram_type} (Persian)")
    plt.grid(True)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close() # Close the plot to free up memory
    return plot_url

def plot_pos_distribution(pos_analysis, title="POS Tag Distribution (Persian)"):
    tags_and_counts = list(pos_analysis['tag_counts'].items())

    sorted_tags = sorted(tags_and_counts, key=lambda item: item[1], reverse=True)

    tags = [item[0] for item in sorted_tags]
    counts = [item[1] for item in sorted_tags]
    descriptions = [pos_analysis['tag_descriptions'].get(tag, tag) for tag in tags]
    reshaped_labels = reshape_persian([f"{tag} ({desc})" for tag, desc in zip(tags, descriptions)])

    ranks = list(range(1, len(tags) + 1))

    plt.figure(figsize=(10, 6))
    plt.plot(ranks, counts, color='darkorange', marker='o', linestyle='-')
    plt.xticks(ranks, reshaped_labels, rotation=45, ha='right')
    plt.xlabel("POS Tag Rank")
    plt.ylabel("Frequency (Counts)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close() # Close the plot to free up memory
    return plot_url

def plot_alphabet_distribution(alpha_dist, title="Persian Alphabet Distribution"):
    """
    Creates a bar chart of the Persian alphabet frequencies.
    """
    if not alpha_dist['counts']:
        print("No alphabet data to plot.")
        return

    chars = list(alpha_dist['counts'].keys())
    counts = list(alpha_dist['counts'].values())

    # Use the reshape_persian helper to ensure characters display correctly
    reshaped_labels = reshape_persian(chars)

    plt.figure(figsize=(12, 6))
    plt.bar(reshaped_labels, counts, color='c') # 'c' for cyan color
    plt.xlabel('Alphabet Characters')
    plt.ylabel('Frequency (Count)')
    plt.title(title)
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close() # Close the plot to free up memory
    return plot_url

if __name__ == "__main__":

    TEXT_FILE_PATH = r"D:\MA\Books\Texts\Kiumars Parsai\The Old Man and the Sea.txt"
    POS_TAGGER_MODEL_PATH = "D:\stylometry\Hazm trained models/pos_tagger.model"
    LEXICONS_DIR_PATH = "D:\stylometry\Json\lexicons_fa"

    try:
        with open(TEXT_FILE_PATH, 'r', encoding='utf-8') as file:
            persian_text = file.read().lstrip('\ufeff')
        print(f"Successfully loaded text from '{TEXT_FILE_PATH}'.")
    except FileNotFoundError:
        print(f"Error: The file '{TEXT_FILE_PATH}' was not found.")
        exit()
    except Exception as e:
        print(f"An error occurred: {e}")
        exit()

    preprocessor = PersianTextPreprocessor(
        pos_tagger_model=POS_TAGGER_MODEL_PATH,
        lexicons_dir=LEXICONS_DIR_PATH
    )

    analysis = preprocessor.comprehensive_text_analysis(persian_text)


    print("============ PERSIAN TEXT ANALYSIS ============")
    print("\nOriginal text:")
    print(persian_text[:100] + "...")  # Show first 100 chars

    print("\n============ BASIC STATISTICS ============")
    print(f"Characters (raw): {analysis['basic_stats']['raw_char_count']}")
    print(f"Words (raw): {analysis['basic_stats']['raw_token_count']}")
    print(f"Characters (processed): {analysis['basic_stats']['processed_char_count']}")
    print(f"Words (processed): {analysis['basic_stats']['processed_token_count']}")


    raw_tokens = preprocessor.tokenize(persian_text)
    processed_tokens = preprocessor.tokenize(preprocessor.clean_text(persian_text))

    print("\n============ TOKENS ============")
    print("Raw tokens (first 15):")
    print(", ".join(raw_tokens[:15]))

    print("Processed tokens (first 15):")
    print(", ".join(processed_tokens[:15]))

    print("\n============ SENTENCE TOKENS ============")
    sentences = preprocessor.sentence_tokenize(persian_text)
    print(f"Total sentences: {len(sentences)}")
    print("Top 20 Sentences:")
    for i, sentence in enumerate(sentences[:10], 1):
      display_sentence = sentence[:100] + "..." if len(sentence) > 100 else sentence
      print(f"Sentence {i}: {display_sentence}")

    print("\n============ WORD LENGTH STATISTICS ============")
    wl = analysis['word_length']
    print(f"Average word length: {wl['mean_word_length']:.2f} characters")
    print(f"Median word length: {wl['median_word_length']:.1f} characters")
    print(f"Word length range: {wl['min_word_length']} to {wl['max_word_length']} characters")
    print("Word length distribution:")
    for length, count in sorted(wl['word_length_distribution'].items()):
        print(f"{length} chars: {count} words")

    print("\n============ SENTENCE STATISTICS ============")
    st = analysis['sentence_length']['by_tokens']
    print(f"Total sentences: {st['sentence_count']}")
    print(f"Average sentence length: {st['mean_sentence_length_tokens']:.2f} words")
    print(f"Median sentence length: {st['median_sentence_length_tokens']:.1f} words")
    print(f"Sentence length range: {st['min_sentence_length_tokens']} to {st['max_sentence_length_tokens']} words")

    print("============ VOCABULARY RICHNESS METRICS ============")
    vr = analysis['vocabulary_richness']
    print(f"Total tokens: {vr['total_tokens']}")
    print(f"Unique tokens: {vr['unique_tokens']} ({vr['unique_tokens']/vr['total_tokens']*100:.2f}% of total)")
    print(f"Type-Token Ratio (TTR): {vr['ttr']:.4f}")
    print(f"Root TTR: {vr['root_ttr']:.4f}")
    print(f"Log TTR: {vr['log_ttr']:.4f}")
    print(f"Hapax Legomena (words occurring once): {vr['hapax_count']} ({vr['hapax_ratio']*100:.2f}% of total)")
    print(f"Examples: {', '.join(vr['hapax_legomena_words'][:10])}")
    print(f"Dis Legomena (words occurring twice): {vr['dis_count']} ({vr['dis_count']/vr['total_tokens']*100:.2f}% of total)")
    print(f"Examples: {', '.join(vr['dis_legomena_words'][:10])}")
    print(f"Yule's K (lexical repetition): {vr['yule_k']:.4f}")

    print("\n============ TOP 100 MOST FREQUENT WORDS ============")
    total_tokens = analysis['vocabulary_richness']['total_tokens']
    if total_tokens > 0:
        for word, count in analysis['word_frequencies']['top_100_words'][:100]:
            percentage = (count / total_tokens) * 100
            print(f"{word} ({count})  ({percentage:.2f}%)")
    else:
        # Fallback to prevent division by zero if there are no tokens
        for word, count in analysis['word_frequencies']['top_100_words'][:100]:
            print(f"{word} ({count})")

    print("============ N-GRAM ANALYSIS ============")
    # Print bigrams
    if 'bigrams' in analysis['ngrams']:
        print("\nTop 10 bigrams:")
        for bigram, count in analysis['ngrams']['bigrams']['top_10']:
            print(f"{bigram}: {count}")

    # Print trigrams
    if 'trigrams' in analysis['ngrams']:
        print("\nTop 10 trigrams:")
        for trigram, count in analysis['ngrams']['trigrams']['top_10']:
            print(f"{trigram}: {count}")

    print("\n============ POS TAGGING ANALYSIS ============")
    pos_analysis = analysis['pos_analysis']

    print(f"Total tokens analyzed: {pos_analysis['total_tokens']}")
    print(f"Unique POS tags found: {pos_analysis['unique_tags']}")

    print("POS Tag Distribution (top tags):")
    for tag, count, description in pos_analysis['most_common'][:10]:
        percentage = pos_analysis['tag_percentages'][tag]
        print(f"{tag} ({description}): {count} tokens ({percentage:.1f}%)")

    print("\nExample words by POS tag:")
    tag_examples = preprocessor.get_pos_tag_examples(persian_text)
    for tag, data in tag_examples.items():
        print(f"{tag} ({data['description']}): {', '.join(data['examples'][:10])}")

    sc = analysis['sentence_complexity']
    print("\n--- Sentence Complexity Analysis (by Verb Count) ---")
    print(f"Total sentences analyzed: {sc['total_sentences']}")
    print(f"  - Simple (1 verb): {sc['simple_sentences']} ({sc['simple_percentage']:.1f}%)")
    print(f"  - Complex (2+ verbs): {sc['complex_sentences']} ({sc['complex_percentage']:.1f}%)")
    print(f"  - No-verb sentences: {sc['no_verb_sentences']} ({sc['no_verb_percentage']:.1f}%)")

    # --- NEW: Print SFG Features Results ---
    sfg = analysis['sfg_features']
    print("\n--- SFG Stylistic Features Analysis ---")
    print(f"Total SFG features found: {sfg['total_features_found']}")
    print("Feature Frequencies (Attribute | Value -> Relative Frequency):")
    # Sort by frequency for better readability
    sorted_sfg = sorted(sfg['feature_relative_frequencies'].items(), key=lambda x: -x[1])
    for (feature, freq) in sorted_sfg[:10]: # Display top 10
        attr, val = feature.split('|')
        print(f"  - {attr:20} | {val:30} | {freq:.4f}")

    print("\n============ PERSIAN ALPHABET DISTRIBUTION ============")
    alpha_dist = analysis['alphabet_distribution']
    if alpha_dist['total_alphabets'] > 0:
        print(f"Total alphabetic characters counted: {alpha_dist['total_alphabets']}")
        print("{:<8} {:<12} {:<10}".format("Char", "Count", "Percent (%)"))
        print("-" * 32)
        for char, count in alpha_dist['counts'].items():
            percentage = alpha_dist['percentages'][char]
            # Reshape for correct printing in the output cell
            reshaped_char = get_display(arabic_reshaper.reshape(char))
            print(f"{reshaped_char:<8} {count:<12} {percentage:<10.2f}")
    else:
        print("No Persian alphabetic characters found.")


    print("\n============ PERSIAN PUNCTUATION DISTRIBUTION ============")
    punc_dist = analysis['punctuation_distribution']
    if punc_dist['total_punctuations'] > 0:
        print(f"Total punctuation characters counted: {punc_dist['total_punctuations']}")
        print("{:<8} {:<12} {:<10}".format("Punc", "Count", "Percent (%)"))
        print("-" * 32)
        for punc, count in punc_dist['counts'].items():
            percentage = punc_dist['percentages'][punc]
            # Reshape for correct printing in the output cell
            reshaped_punc = get_display(arabic_reshaper.reshape(punc))
            print(f"{reshaped_punc:<8} {count:<12} {percentage:<10.2f}")
    else:
        print("No Persian punctuation characters found.")


    # Visualize word length distribution
    plot_word_length_distribution(analysis['word_length']['word_length_distribution'], title="Mendenhall's Characteristic Curve")

    # Visualize sentence length distribution (by tokens)
    plot_sentence_length_distribution(analysis['sentence_length']['by_tokens']['sentence_length_distribution'], unit='tokens')

    # Vocabulary richness
    plot_vocabulary_richness(analysis['vocabulary_richness'])

    # Word frequency distribution
    plot_word_frequency_distribution(analysis['word_frequencies']['top_100_words'][:20])

    # N-gram frequencies (
    plot_ngram_frequencies(analysis['ngrams']['unigrams']['top_10'], "Unigrams")
    plot_ngram_frequencies(analysis['ngrams']['bigrams']['top_10'], "Bigrams")
    plot_ngram_frequencies(analysis['ngrams']['trigrams']['top_10'], "Trigrams")

    # POS tag distribution
    plot_pos_distribution(analysis['pos_analysis'])

    # Visualize Alphabet Distribution
    plot_alphabet_distribution(analysis['alphabet_distribution'])