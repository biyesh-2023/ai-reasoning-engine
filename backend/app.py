from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from dotenv import load_dotenv
import re
import json
from typing import List, Dict, Any
import requests
import math

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

# Try to import sympy, but provide fallback if not available
try:
    import sympy as sp
    from sympy import symbols, solve, simplify
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    print("⚠️ sympy not available - equation solving will be limited")

# Serve frontend
@app.route('/')
def serve_frontend():
    return send_from_directory('../frontend', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('../frontend', path)

# ============================================
# SIMPLIFIED GROQ CLIENT
# ============================================
class SimpleGroqClient:
    def __init__(self):
        self.api_key = os.getenv('GROQ_API_KEY')
        if not self.api_key:
            print("⚠️ GROQ_API_KEY not found - some features will be limited")
            self.api_key = "dummy-key-for-development"
        
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_response(self, messages: List[Dict], max_tokens: int = 400, temperature: float = 0.7) -> str:
        try:
            # If no API key, return mock response
            if not self.api_key or self.api_key == "dummy-key-for-development":
                return self._get_mock_response(messages[0]['content'])
            
            payload = {
                "model": "llama-3.1-8b-instant",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9
            }
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content'].strip()
            else:
                return f"Error: API returned {response.status_code}. Please check your GROQ_API_KEY."
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _get_mock_response(self, prompt: str) -> str:
        """Provide mock responses when no API key is available"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['math', 'calculate', 'solve', 'equation']):
            return "42"  # The answer to everything
        elif any(word in prompt_lower for word in ['explain', 'what is']):
            return "This is a comprehensive explanation of the topic you asked about. In a real deployment with a GROQ_API_KEY, you would get a detailed AI-generated response."
        else:
            return "I understand your question. With a proper GROQ_API_KEY environment variable set, I would provide a detailed AI-generated response to your query."

# ============================================
# SIMPLE PROBLEM SOLVER - NO INTERNAL STEPS
# ============================================
class SimpleProblemSolver:
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def solve_problem(self, question: str) -> Dict[str, Any]:
        try:
            # Generate direct, simple answer without internal reasoning steps
            prompt = f"""Please provide a clear, direct answer to the following question. 
            Do not show any internal reasoning steps like "Step 1: Analyze" or "Step 2: Apply reasoning".
            Just provide the answer in a natural, conversational way.
            
            If the answer involves multiple steps or categories, structure it with clear paragraphs or bullet points.
            But do not number the steps artificially.
            
            Question: {question}
            
            Answer:"""
            
            messages = [{"role": "user", "content": prompt}]
            final_answer = self.llm.generate_response(messages, max_tokens=600)
            
            return {
                'final_answer': final_answer,
                'subproblems': [],
                'decomposition_steps': [],
                'question_type': 'direct',
                'has_steps': False
            }
            
        except Exception as e:
            return self._handle_error(question, str(e))
    
    def _handle_error(self, question: str, error: str) -> Dict[str, Any]:
        """Handle errors gracefully"""
        return {
            'final_answer': f"I encountered an error while processing your question: {error}. Please try again.",
            'subproblems': [],
            'decomposition_steps': [],
            'question_type': 'error',
            'has_steps': False
        }

# ============================================
# SIMPLE EXECUTOR FOR MATH
# ============================================
class SimpleCalculator:
    def solve(self, text: str) -> str:
        try:
            # Extract mathematical expressions
            expressions = re.findall(r'[\d+\-*/().^]+', text)
            for expr in expressions:
                try:
                    clean_expr = expr.replace('^', '**')
                    result = eval(clean_expr, {"__builtins__": None}, {
                        "sin": math.sin, "cos": math.cos, "tan": math.tan,
                        "sqrt": math.sqrt, "log": math.log, "exp": math.exp,
                        "pi": math.pi, "e": math.e
                    })
                    return f"{expr} = {result}"
                except:
                    continue
            return "I can help with calculations. Please provide a clear math expression."
        except Exception as e:
            return f"Calculation error: {str(e)}"

# ============================================
# SIMPLE AGENTIC SYSTEM
# ============================================
class SimpleAgenticSystem:
    def __init__(self):
        self.llm = SimpleGroqClient()
        self.solver = SimpleProblemSolver(self.llm)
        self.calculator = SimpleCalculator()
    
    def solve_problem(self, question: str) -> Dict[str, Any]:
        # Check if it's a pure math problem
        if re.search(r'[\d+\-*/().^=]', question) and not re.search(r'[a-zA-Z]', question.replace(' ', '')):
            math_result = self.calculator.solve(question)
            return {
                'final_answer': math_result,
                'subproblems': [],
                'decomposition_steps': [],
                'question_type': 'math',
                'has_steps': False
            }
        else:
            return self.solver.solve_problem(question)

# ============================================
# API ENDPOINTS
# ============================================
try:
    agent = SimpleAgenticSystem()
    print("✓ Simple Agentic System initialized successfully")
    print(f"✓ Sympy available: {SYMPY_AVAILABLE}")
except Exception as e:
    print(f"✗ Failed to initialize Agentic System: {e}")
    agent = None

@app.route('/api/solve', methods=['POST'])
def solve():
    try:
        if agent is None:
            return jsonify({
                'error': 'System not properly initialized.'
            }), 500
        
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'Please enter a question'}), 400
        
        # Handle very short questions and greetings
        if len(question) < 3:
            return jsonify({
                'final_answer': "I'd be happy to help! Please ask me a specific question.",
                'subproblems': [],
                'decomposition_steps': [],
                'question_type': 'general',
                'has_steps': False
            })
        
        # Handle common greetings
        greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
        if question.lower().strip('.!?') in greetings:
            return jsonify({
                'final_answer': "Hello! I'm an AI Assistant. I can help you with questions, explanations, calculations, and problem solving. What would you like to know?",
                'subproblems': [],
                'decomposition_steps': [],
                'question_type': 'greeting',
                'has_steps': False
            })
        
        # Process the question
        result = agent.solve_problem(question)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'final_answer': 'Sorry, I encountered an error processing your question. Please try again.'
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    status = 'healthy' if agent else 'unhealthy'
    return jsonify({
        'status': status,
        'model': 'groq-llama-3.1',
        'system': 'simple-ai-assistant',
        'sympy_available': SYMPY_AVAILABLE
    })

@app.route('/api/test-key', methods=['GET'])
def test_key():
    api_key = os.getenv('GROQ_API_KEY', '')
    if not api_key:
        return jsonify({'error': 'API key not found'}), 500
    
    masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "invalid"
    return jsonify({
        'status': 'API key loaded',
        'key_preview': masked_key,
        'key_length': len(api_key),
        'system': 'Simple AI Assistant'
    })

@app.route('/api/capabilities', methods=['GET'])
def capabilities():
    return jsonify({
        'capabilities': [
            'Simple Q&A and explanations',
            'Mathematical calculations',
            'Direct, clear answers without internal steps'
        ]
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug_mode = os.getenv('DEBUG', 'false').lower() == 'true'
    
    print("🚀 Simple AI Assistant Starting...")
    print(f"🌐 Server will run on port: {port}")
    print("✅ System ready to provide direct, clear answers!")
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)