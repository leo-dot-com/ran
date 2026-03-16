from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import tempfile
import os
import json
from datetime import datetime
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load Whisper model (load once at startup)
model = None

def load_model():
    global model
    if model is None:
        logger.info("Loading Whisper model...")
        model = whisper.load_model("tiny")
        logger.info("Whisper model loaded successfully")
    return model

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/analyze-ran', methods=['POST'])
def analyze_ran():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        test_type = request.form.get('test_type', 'colors')
        expected_items = request.form.get('expected_items', '')

        if not expected_items:
            return jsonify({'error': 'Expected items list is required'}), 400

        # Load expected items
        expected_list = json.loads(expected_items)

        # Save audio to temporary file (preserve original extension)
        ext = os.path.splitext(audio_file.filename)[1] or '.webm'
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_audio:
            audio_file.save(temp_audio.name)
            temp_path = temp_audio.name

        try:
            # Load model and transcribe
            model = load_model()

            # Transcribe with word-level timestamps
            result = model.transcribe(
                temp_path,
                word_timestamps=True,
                fp16=False
            )

            # Extract words from segments, trimming whitespace
            words = []
            for segment in result.get('segments', []):
                for word_info in segment.get('words', []):
                    words.append({
                        'word': word_info['word'].strip(),
                        'start': word_info['start'],
                        'end': word_info['end']
                    })

            # Compute total duration from the last word's end, or from segments, or fallback to 0
            total_duration = 0
            if words:
                total_duration = words[-1]['end']
            elif result.get('segments'):
                total_duration = result['segments'][-1]['end']
            else:
                total_duration = result.get('duration', 0)

            # If no word timestamps were generated, create synthetic ones
            if not words and result['text'].strip():
                logger.info("No word timestamps from Whisper, generating synthetic ones")
                text_words = result['text'].split()
                if text_words and total_duration > 0:
                    seg_duration = total_duration / len(text_words)
                    words = []
                    for i, w in enumerate(text_words):
                        words.append({
                            'word': w.strip(),
                            'start': i * seg_duration,
                            'end': (i + 1) * seg_duration
                        })

            # Process results
            analysis = analyze_ran_performance(words, result['text'], total_duration, expected_list, test_type)

            return jsonify({
                'success': True,
                'transcription': result['text'],
                'words': words,
                'analysis': analysis,
                'test_type': test_type
            })

        finally:
            # Clean up temporary file
            os.unlink(temp_path)

    except Exception as e:
        logger.error(f"Error processing RAN analysis: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Processing failed: {str(e)}'
        }), 500

def analyze_ran_performance(words, text, total_duration, expected_items, test_type):
    """Analyze RAN performance based on word timestamps and expected items"""

    if not words:
        # No words at all – return zero metrics
        return {
            'total_time': total_duration,
            'items_correct': 0,
            'total_items': len(expected_items),
            'accuracy': 0,
            'average_response_time': 0,
            'response_times': [],
            'hesitations': 0,
            'self_corrections': 0,
            'errors': len(expected_items),
            'naming_speed_wps': 0,
            'consistency_score': 0,
            'ran_score': 0,
        }

    # Prepare expected list (lowercase)
    expected_lower = [item.lower() for item in expected_items]
    total_items = len(expected_lower)

    # Extract transcribed words (lowercase, stripped)
    transcribed_words = [w['word'].lower() for w in words]

    # Calculate accuracy and collect response times
    correct_count = 0
    errors = 0
    self_corrections = 0
    response_times = []

    for i, word_info in enumerate(words):
        word = word_info['word'].lower()
        start_time = word_info['start']
        end_time = word_info['end']

        # Response time: time from previous word end to current word start
        if i > 0:
            prev_end = words[i-1]['end']
            response_time = start_time - prev_end
            response_times.append(response_time)

        # Compare with expected (position‑based)
        if i < total_items:
            expected_word = expected_lower[i]
            if word == expected_word:
                correct_count += 1
            else:
                errors += 1
                # Check if next word is a self‑correction
                if i + 1 < len(words) and words[i+1]['word'].lower() == expected_word:
                    self_corrections += 1

    # Hesitations: response times > 1 second
    hesitations = sum(1 for rt in response_times if rt > 1.0)

    # Naming speed (words per second)
    naming_speed_wps = len(words) / total_duration if total_duration > 0 else 0

    # Accuracy percentage
    accuracy = (correct_count / total_items) * 100 if total_items > 0 else 0

    # Average response time
    avg_response_time = np.mean(response_times) if response_times else 0

    # Composite scores
    consistency_score = max(0, 100 - (hesitations * 10 + errors * 5))
    ran_score = (accuracy * 0.4) + (consistency_score * 0.3) + (max(0, 100 - (avg_response_time * 20)) * 0.3)

    return {
        'total_time': total_duration,
        'items_correct': correct_count,
        'total_items': total_items,
        'accuracy': accuracy,
        'average_response_time': avg_response_time,
        'response_times': response_times,
        'hesitations': hesitations,
        'self_corrections': self_corrections,
        'errors': errors,
        'naming_speed_wps': naming_speed_wps,
        'consistency_score': consistency_score,
        'ran_score': ran_score,
    }

if __name__ == '__main__':
    # Pre-load model when starting up
    load_model()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
