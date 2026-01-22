import pandas as pd
import re
import random
import os
import json
from datetime import datetime
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, render_template
import nltk
from nltk.corpus import stopwords

# Download stopwords if not present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# ---------------- CONFIG ----------------
app = Flask(__name__)
app.secret_key = 'enhanced-chatbot-secret-key-2024'
DATA_FILE = 'faq_dataset.csv'
LOG_FILE = 'chat_logs.csv'
UNKNOWN_FILE = 'unknown_questions.csv'
SIMILARITY_THRESHOLD = 0.30
LEARNING_MODE = True  # Toggle for learning mode

# ---------------- GLOBALS ----------------
faq_data = None
vectorizer = None
tfidf_matrix = None
chat_history = []
unknown_questions = []
stop_words = set(stopwords.words('english'))

# Enhanced fallback responses
fallback_en = [
    "Hmm, I don't have that information yet 🤔 Can you try rephrasing?",
    "Good question! I'm still learning about that topic.",
    "I couldn't find an exact match. Would you like me to connect you with a human advisor?",
    "That's an interesting question! Let me check my resources...",
    "I'm not sure about that one. Try asking about university services, policies, or procedures.",
    "I'm still expanding my knowledge base. Could you ask something about registration, exams, or campus facilities?"
]

fallback_ar = [
    "همم، لا أملك هذه المعلومة بعد 🤔 هل يمكنك إعادة صياغة السؤال؟",
    "سؤال جيد! لا زلت أتعلم عن هذا الموضوع.",
    "لم أجد إجابة دقيقة. هل تريدني أن أوصلك بمستشار بشري؟",
    "هذا سؤال مثير للاهتمام! دعني أتحقق من مصادر المعلومات...",
    "لست متأكداً من هذا. حاول أن تسأل عن خدمات الجامعة، السياسات، أو الإجراءات.",
    "لا زلت أوسع قاعدة معرفتي. هل يمكنك السؤال عن التسجيل، الامتحانات، أو مرافق الحرم الجامعي؟"
]

# ---------------- HELPER FUNCTIONS ----------------
def is_arabic(text):
    """Detect if text contains Arabic characters"""
    return any('\u0600' <= c <= '\u06FF' for c in text)

def enhanced_clean_text(text):
    """Advanced text cleaning with stopword removal"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Tokenize and remove stopwords
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    
    return ' '.join(filtered_words)

def log_chat_interaction(user_query, bot_response, confidence, category=None):
    """Log all chat interactions for analytics"""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lang = 'AR' if is_arabic(user_query) else 'EN'
        
        log_entry = {
            'timestamp': timestamp,
            'user_query': user_query,
            'bot_response': bot_response,
            'language': lang,
            'confidence': float(confidence),
            'category': category or 'General',
            'response_type': 'Answered' if confidence >= SIMILARITY_THRESHOLD else 'Fallback'
        }
        
        # Add to session history for admin page
        chat_history.append(log_entry)
        
        # Keep only last 100 entries in memory
        if len(chat_history) > 100:
            chat_history.pop(0)
        
        # Save to CSV file
        log_df = pd.DataFrame([log_entry])
        file_exists = os.path.exists(LOG_FILE)
        log_df.to_csv(LOG_FILE, mode='a', header=not file_exists, index=False, encoding='utf-8')
        
        return True
    except Exception as e:
        print(f"Logging error: {e}")
        return False

def save_unknown_question(user_query):
    """Save unknown questions for review (Learning Mode)"""
    if LEARNING_MODE:
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            unknown_entry = {
                'timestamp': timestamp,
                'question': user_query,
                'language': 'AR' if is_arabic(user_query) else 'EN',
                'reviewed': False
            }
            
            unknown_questions.append(unknown_entry)
            
            # Keep only last 50 unknown questions in memory
            if len(unknown_questions) > 50:
                unknown_questions.pop(0)
            
            unknown_df = pd.DataFrame([unknown_entry])
            file_exists = os.path.exists(UNKNOWN_FILE)
            unknown_df.to_csv(UNKNOWN_FILE, mode='a', header=not file_exists, index=False, encoding='utf-8')
            
            print(f"📝 Learning Mode: Saved unknown question: {user_query}")
            return True
        except Exception as e:
            print(f"Error saving unknown question: {e}")
            return False
    return False

def get_category(query):
    """Categorize queries based on keywords"""
    categories = {
        'Registration': ['register', 'course', 'enroll', 'enrollment', 'add/drop', 'registration'],
        'Technical': ['password', 'reset', 'email', 'login', 'portal', 'technical', 'software', 'it'],
        'Academic': ['exam', 'schedule', 'grade', 'transcript', 'advisor', 'major', 'course', 'study', 'academic'],
        'Facilities': ['library', 'hours', 'location', 'campus', 'building', 'lab', 'facility', 'room'],
        'Support': ['help', 'support', 'contact', 'assistance', 'helpdesk'],
        'Financial': ['scholarship', 'financial aid', 'fee', 'tuition', 'payment', 'financial'],
        'Student Life': ['club', 'event', 'workshop', 'activity', 'dormitory', 'housing', 'life', 'student']
    }
    
    query_lower = query.lower()
    for category, keywords in categories.items():
        if any(keyword in query_lower for keyword in keywords):
            return category
    
    return 'General'

def get_analytics():
    """Generate analytics from chat logs"""
    analytics = {
        'total_questions': 0,
        'answered_questions': 0,
        'fallback_count': 0,
        'english_questions': 0,
        'arabic_questions': 0,
        'avg_confidence': 0,
        'top_categories': {},
        'unknown_count': 0
    }
    
    try:
        # Get from chat history
        if chat_history:
            analytics['total_questions'] = len(chat_history)
            analytics['answered_questions'] = len([c for c in chat_history if c['response_type'] == 'Answered'])
            analytics['fallback_count'] = len([c for c in chat_history if c['response_type'] == 'Fallback'])
            analytics['english_questions'] = len([c for c in chat_history if c['language'] == 'EN'])
            analytics['arabic_questions'] = len([c for c in chat_history if c['language'] == 'AR'])
            
            confidences = [c['confidence'] for c in chat_history if 'confidence' in c]
            if confidences:
                analytics['avg_confidence'] = round(sum(confidences) / len(confidences), 3)
            
            # Count categories
            categories = [c.get('category', 'General') for c in chat_history]
            category_counts = Counter(categories)
            analytics['top_categories'] = dict(category_counts.most_common(5))
        
        # Get unknown questions count
        if os.path.exists(UNKNOWN_FILE):
            try:
                unknown_df = pd.read_csv(UNKNOWN_FILE, encoding='utf-8')
                analytics['unknown_count'] = len(unknown_df)
            except:
                analytics['unknown_count'] = len(unknown_questions)
        else:
            analytics['unknown_count'] = len(unknown_questions)
            
    except Exception as e:
        print(f"Analytics error: {e}")
    
    return analytics

def load_unknown_questions():
    """Load unknown questions from file"""
    global unknown_questions
    unknown_questions = []
    
    if os.path.exists(UNKNOWN_FILE):
        try:
            unknown_df = pd.read_csv(UNKNOWN_FILE, encoding='utf-8')
            unknown_questions = unknown_df.to_dict('records')
        except Exception as e:
            print(f"Error loading unknown questions: {e}")
            unknown_questions = []

def load_chat_history():
    """Load chat history from file"""
    global chat_history
    chat_history = []
    
    if os.path.exists(LOG_FILE):
        try:
            logs_df = pd.read_csv(LOG_FILE, encoding='utf-8')
            chat_history = logs_df.to_dict('records')
            
            # Keep only last 100 entries
            if len(chat_history) > 100:
                chat_history = chat_history[-100:]
                
        except Exception as e:
            print(f"Error loading chat logs: {e}")
            chat_history = []

# ---------------- MODEL LOADING ----------------
def load_and_prepare_model():
    global faq_data, vectorizer, tfidf_matrix

    if not os.path.exists(DATA_FILE):
        print(f"❌ FAQ dataset not found at: {os.path.abspath(DATA_FILE)}")
        print(f"Current directory: {os.getcwd()}")
        return False

    try:
        faq_data = pd.read_csv(DATA_FILE, encoding='utf-8')

        if 'Question' not in faq_data.columns or 'Answer' not in faq_data.columns:
            print("❌ CSV must contain Question and Answer columns.")
            return False

        # Add category if not present
        if 'Category' not in faq_data.columns:
            faq_data['Category'] = faq_data['Question'].apply(get_category)

        faq_data['clean_question'] = faq_data['Question'].apply(enhanced_clean_text)

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(faq_data['clean_question'])

        print(f"✅ Model loaded. {len(faq_data)} questions indexed.")
        print(f"📊 Categories: {faq_data['Category'].unique().tolist()}")
        return True

    except Exception as e:
        print("❌ Model loading error:", e)
        return False

# ---------------- CORE CHATBOT LOGIC ----------------
def get_best_response(user_query):
    if not user_query or not isinstance(user_query, str) or not user_query.strip():
        return "Please enter a valid question.", 0, [], 'General'

    lang_ar = is_arabic(user_query)
    cleaned_query = enhanced_clean_text(user_query)
    category = get_category(user_query)

    try:
        if vectorizer is None or tfidf_matrix is None:
            return "System not ready. Please try again.", 0, [], category

        query_vec = vectorizer.transform([cleaned_query])
        similarities = cosine_similarity(query_vec, tfidf_matrix)[0]
        
        best_index = similarities.argmax()
        best_score = float(similarities[best_index])
        
        # Get top 3 matches for suggestions
        top_indices = similarities.argsort()[-3:][::-1]
        suggestions = []
        
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum threshold for suggestions
                suggestions.append({
                    'question': faq_data.iloc[idx]['Question'],
                    'similarity': float(similarities[idx])
                })

        print(f"Query: '{user_query}' | Score: {best_score:.3f} | Category: {category}")

        if best_score >= SIMILARITY_THRESHOLD:
            response = str(faq_data.iloc[best_index]['Answer'])
            log_chat_interaction(user_query, response, best_score, category)
            return response, best_score, [], category
        else:
            # Get suggestions if confidence is between 0.1 and threshold
            did_you_mean = []
            if 0.1 <= best_score < SIMILARITY_THRESHOLD:
                for idx in top_indices[:3]:
                    if similarities[idx] > 0.1:
                        did_you_mean.append(str(faq_data.iloc[idx]['Question']))
            
            # Save unknown question for learning
            save_unknown_question(user_query)
            
            # Choose fallback response
            if lang_ar:
                response = random.choice(fallback_ar)
            else:
                response = random.choice(fallback_en)
            
            # Add suggestions to response if available
            if did_you_mean:
                if lang_ar:
                    response += "\n\nهل تقصد أحد هذه الأسئلة؟\n"
                    for i, q in enumerate(did_you_mean, 1):
                        response += f"{i}. {q}\n"
                else:
                    response += "\n\nDid you mean one of these?\n"
                    for i, q in enumerate(did_you_mean, 1):
                        response += f"{i}. {q}\n"
            
            log_chat_interaction(user_query, response, best_score, category)
            return response, best_score, did_you_mean, category

    except Exception as e:
        print("❌ Runtime error:", e)
        error_msg = "⚠️ Internal processing error. Please try again."
        log_chat_interaction(user_query, error_msg, 0, category)
        return error_msg, 0, [], category

# ---------------- ROUTES ----------------
@app.route('/')
def index():
    """Main chat interface"""
    return render_template('index.html', learning_mode=LEARNING_MODE)

@app.route('/admin')
def admin_dashboard():
    """Admin analytics dashboard"""
    analytics = get_analytics()
    load_unknown_questions()
    load_chat_history()
    
    # Get recent chats (last 20)
    recent_chats = chat_history[-20:] if chat_history else []
    
    return render_template('admin.html', 
                         analytics=analytics, 
                         unknown_questions=unknown_questions,
                         recent_chats=recent_chats,
                         learning_mode=LEARNING_MODE)

@app.route('/get_response', methods=['POST'])
def get_response_route():
    """API endpoint for chat responses"""
    try:
        data = request.get_json(force=True)
        user_query = data.get('query', '')
        
        if not user_query:
            return jsonify({
                'response': 'Please enter a question.',
                'confidence': 0,
                'suggestions': [],
                'category': 'General',
                'learning_mode': LEARNING_MODE
            })

        response, confidence, suggestions, category = get_best_response(user_query)
        
        return jsonify({
            'response': response,
            'confidence': confidence,
            'suggestions': suggestions,
            'category': category,
            'learning_mode': LEARNING_MODE
        })
    except Exception as e:
        print(f"API error: {e}")
        return jsonify({
            'response': 'Server error. Please try again.',
            'confidence': 0,
            'suggestions': [],
            'category': 'Error',
            'learning_mode': LEARNING_MODE
        })

@app.route('/toggle_learning', methods=['POST'])
def toggle_learning():
    """Toggle learning mode"""
    global LEARNING_MODE
    LEARNING_MODE = not LEARNING_MODE
    return jsonify({
        'success': True,
        'learning_mode': LEARNING_MODE,
        'message': f"Learning mode {'enabled' if LEARNING_MODE else 'disabled'}"
    })

@app.route('/get_stats', methods=['GET'])
def get_stats():
    """Get real-time statistics"""
    analytics = get_analytics()
    return jsonify(analytics)

@app.route('/get_categories', methods=['GET'])
def get_categories():
    """Get all available categories"""
    if faq_data is not None and 'Category' in faq_data.columns:
        categories = faq_data['Category'].unique().tolist()
        return jsonify({'categories': categories})
    return jsonify({'categories': []})

@app.route('/questions_by_category/<category>', methods=['GET'])
def questions_by_category(category):
    """Get questions by category"""
    if faq_data is not None:
        filtered = faq_data[faq_data['Category'].str.contains(category, case=False, na=False)]
        questions = filtered[['Question', 'Answer']].to_dict('records')
        return jsonify({'questions': questions})
    return jsonify({'questions': []})

# ---------------- START ----------------
if __name__ == '__main__':
    print("\n" + "="*50)
    print("FAQ CHATBOT BY ABDALLA ELSHEMALY AND MOHAMMAD KHAIRALLAH")
    print("="*50)
    
    # Load data
    load_unknown_questions()
    load_chat_history()
    
    if load_and_prepare_model():
        print("\n🎯 Features Enabled:")
        print("  ✅ Smart Fallback Responses")
        print("  ✅ Bilingual Support (EN/AR)")
        print("  ✅ 'Did You Mean?' Suggestions")
        print(f"  ✅ Learning Mode: {'ACTIVE' if LEARNING_MODE else 'INACTIVE'}")
        print("  ✅ Admin Analytics Dashboard")
        print("  ✅ Text Cleaning Pipeline")
        print("  ✅ Category-Based Filtering")
        print("  ✅ Confidence Scoring")
        
        print(f"\n📊 Loaded: {len(chat_history)} chat logs, {len(unknown_questions)} unknown questions")
        
        print("\n📊 Access Points:")
        print("  • Chat Interface: http://localhost:5000")
        print("  • Admin Dashboard: http://localhost:5000/admin")
        print("  • API Endpoint: http://localhost:5000/get_response")
        
        print("\n--- Flask Server Running ---\n")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n❌ SERVER FAILED TO START - Check error messages above\n")

        # Add this near the top of app.py
arabic_to_english_map = {
    "كيف أستعيد كلمة المرور؟": "How do I reset my university email password?",
    "كيف يمكنني التسجيل في الدورات؟": "How do I register for courses online?",
    "أين المكتبة؟": "Where is the main university library located?",
    "ما هي ساعات العمل؟": "What are the library's opening and closing hours?",
    "كيف أحصل على المساعدة المالية؟": "How do I apply for scholarships or financial aid?",
    "كيف أتواصل مع مستشاري الأكاديمي؟": "How can I contact my academic advisor?",
    "ما هي مواعيد الامتحانات؟": "How do I check my exam schedule and results?",
    "كيف أبلغ عن سلوك غير لائق؟": "How do I report academic misconduct or harassment?",
    "أين يمكنني العثور على التقويم الأكاديمي؟": "Where can I find the academic calendar?",
    "كيف أنضم إلى النوادي الطلابية؟": "How do I join student clubs and organizations?",
}

def load_and_prepare_model():
    global faq_data, vectorizer_en, vectorizer_ar, tfidf_matrix_en, tfidf_matrix_ar

    if not os.path.exists(DATA_FILE):
        print(f"❌ FAQ dataset not found at: {os.path.abspath(DATA_FILE)}")
        return False

    try:
        faq_data = pd.read_csv(DATA_FILE, encoding='utf-8')

        if 'Question' not in faq_data.columns or 'Answer' not in faq_data.columns:
            print("❌ CSV must contain Question and Answer columns.")
            return False

        # Add category if not present
        if 'Category' not in faq_data.columns:
            faq_data['Category'] = faq_data['Question'].apply(get_category)
        
        # Clean English questions
        faq_data['clean_question_en'] = faq_data['Question'].apply(enhanced_clean_text) 
        
        if has_arabic_questions:
            # Clean Arabic questions (different cleaning might be needed)
            faq_data['clean_question_ar'] = faq_data['Question_ar'].apply(clean_arabic_text)
        
        # Create English vectorizer
        vectorizer_en = TfidfVectorizer()
        tfidf_matrix_en = vectorizer_en.fit_transform(faq_data['clean_question_en'])
        
        # Create Arabic vectorizer if Arabic questions exist
        if has_arabic_questions and 'clean_question_ar' in faq_data.columns:
            vectorizer_ar = TfidfVectorizer()
            tfidf_matrix_ar = vectorizer_ar.fit_transform(faq_data['clean_question_ar'])
        else:
            vectorizer_ar = None
            tfidf_matrix_ar = None

        print(f"✅ Model loaded. {len(faq_data)} questions indexed.")
        if has_arabic_questions:
            print(f"🌍 Bilingual model: English and Arabic support enabled")
        return True

    except Exception as e:
        print("❌ Model loading error:", e)
        return False