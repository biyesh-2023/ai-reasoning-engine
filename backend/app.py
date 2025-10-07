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
# SIMPLIFIED PROBLEM SOLVER
# ============================================
class SimpleProblemSolver:
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def needs_step_by_step(self, question: str) -> bool:
        """Determine if the question needs step-by-step solution"""
        question_lower = question.lower()
        
        # Questions that typically need step-by-step
        step_questions = [
            'solve', 'calculate', 'compute', 'step by step', 'show steps',
            'how to', 'process', 'method', 'equation', 'formula', 'math',
            'proof', 'derivation'
        ]
        
        # Questions that typically don't need step-by-step
        simple_questions = [
            'who is', 'what is', 'explain', 'describe', 'tell me about',
            'define', 'meaning of', 'when did', 'where is'
        ]
        
        if any(word in question_lower for word in step_questions):
            return True
        if any(word in question_lower for word in simple_questions):
            return False
        
        # Default to simple answers
        return False
    
    def solve_problem(self, question: str) -> Dict[str, Any]:
        try:
            needs_steps = self.needs_step_by_step(question)
            
            if needs_steps:
                return self._solve_with_steps(question)
            else:
                return self._solve_simple(question)
                
        except Exception as e:
            return self._handle_error(question, str(e))
    
    def _solve_simple(self, question: str) -> Dict[str, Any]:
        """Provide direct, simple answers"""
        prompt = f"""Please provide a clear, direct answer to the following question. 
        Use simple language and be concise. If the answer has natural steps or categories, 
        you can structure it with bullet points or short paragraphs, but don't show artificial steps like "Step 1: Analyze".

        Question: {question}

        Answer:"""
        
        messages = [{"role": "user", "content": prompt}]
        answer = self.llm.generate_response(messages, max_tokens=500)
        
        return {
            'final_answer': answer,
            'subproblems': [],
            'decomposition_steps': [],
            'question_type': 'simple',
            'needs_steps': False
        }
    
    def _solve_with_steps(self, question: str) -> Dict[str, Any]:
        """Provide step-by-step solutions for complex problems"""
        prompt = f"""Please solve this problem step by step. Show your reasoning process naturally.
        Use clear steps with explanations, but don't use artificial language like "Step 1: Analyze".

        Question: {question}

        Provide a step-by-step solution:"""
        
        messages = [{"role": "user", "content": prompt}]
        answer = self.llm.generate_response(messages, max_tokens=600)
        
        # Extract steps from the answer
        steps = self._extract_steps_from_answer(answer)
        
        return {
            'final_answer': answer,
            'subproblems': steps,
            'decomposition_steps': [],
            'question_type': 'complex',
            'needs_steps': True
        }
    
    def _extract_steps_from_answer(self, answer: str) -> List[Dict]:
        """Extract natural steps from the answer text"""
        steps = []
        lines = answer.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or 
                        line[0].isdigit and '. ' in line[:5] or
                        'step' in line.lower()[:10]):
                steps.append({
                    'subproblem': line,
                    'tool': 'reasoner',
                    'solution': line
                })
        
        return steps if steps else [{
            'subproblem': 'Solution',
            'tool': 'reasoner', 
            'solution': answer
        }]
    
    def _handle_error(self, question: str, error: str) -> Dict[str, Any]:
        """Handle errors gracefully"""
        return {
            'final_answer': f"I encountered an error while processing your question: {error}. Please try again.",
            'subproblems': [],
            'decomposition_steps': [],
            'question_type': 'error',
            'needs_steps': False
        }

# ============================================
# SIMPLIFIED EXECUTORS FOR MATH
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

class SimpleExecutor:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.calculator = SimpleCalculator()
    
    def execute(self, problem: str) -> str:
        # For now, use LLM for most problems, calculator for pure math
        if re.search(r'[\d+\-*/().^=]', problem) and not re.search(r'[a-zA-Z]', problem.replace(' ', '')):
            return self.calculator.solve(problem)
        else:
            messages = [{"role": "user", "content": f"Solve: {problem}"}]
            return self.llm.generate_response(messages, max_tokens=300)

# ============================================
# SIMPLIFIED AGENTIC SYSTEM
# ============================================
class SimpleAgenticSystem:
    def __init__(self):
        self.llm = SimpleGroqClient()
        self.solver = SimpleProblemSolver(self.llm)
        self.executor = SimpleExecutor(self.llm)
    
    def solve_problem(self, question: str) -> Dict[str, Any]:
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
                'needs_steps': False
            })
        
        # Handle common greetings
        greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
        if question.lower().strip('.!?') in greetings:
            return jsonify({
                'final_answer': "Hello! I'm an AI Reasoning Engine. I can help you with questions, explanations, calculations, and problem solving. What would you like to know?",
                'subproblems': [],
                'decomposition_steps': [],
                'question_type': 'greeting',
                'needs_steps': False
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
        'system': 'simple-reasoning-engine',
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
        'system': 'Simple Reasoning Engine'
    })

@app.route('/api/capabilities', methods=['GET'])
def capabilities():
    return jsonify({
        'capabilities': [
            'Simple Q&A and explanations',
            'Mathematical calculations',
            'Step-by-step problem solving when needed',
            'Clear, direct answers'
        ]
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug_mode = os.getenv('DEBUG', 'false').lower() == 'true'
    
    print("🚀 Simple AI Reasoning Engine Starting...")
    print(f"🌐 Server will run on port: {port}")
    print("✅ System ready to answer questions clearly and simply!")
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)