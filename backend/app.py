from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from dotenv import load_dotenv
import re
import json
from typing import List, Dict, Any
import requests
import math
import sympy as sp
from sympy import symbols, solve, simplify

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
# ENHANCED GROQ CLIENT
# ============================================
class EnhancedGroqClient:
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
# ENHANCED PROBLEM DECOMPOSER
# ============================================
class EnhancedProblemDecomposer:
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def analyze_question_type(self, question: str) -> str:
        """Analyze the question to determine its type"""
        question_lower = question.lower()
        
        # Mathematical questions
        math_keywords = ['calculate', 'compute', 'solve', 'equation', 'formula', 'math', 'number', 'digit']
        if any(keyword in question_lower for keyword in math_keywords) or re.search(r'\d+[\+\-\*\/\^]', question):
            return "mathematical"
        
        # Scientific questions
        science_keywords = ['physics', 'chemistry', 'biology', 'scientific', 'experiment', 'theory']
        if any(keyword in question_lower for keyword in science_keywords):
            return "scientific"
        
        # Historical questions
        history_keywords = ['history', 'historical', 'past', 'century', 'war', 'king', 'queen', 'ancient']
        if any(keyword in question_lower for keyword in history_keywords):
            return "historical"
        
        # Geographical questions
        geo_keywords = ['country', 'city', 'capital', 'river', 'mountain', 'continent', 'map', 'location']
        if any(keyword in question_lower for keyword in geo_keywords):
            return "geographical"
        
        # Logical reasoning
        logic_keywords = ['logic', 'reasoning', 'if then', 'therefore', 'implies', 'deduce']
        if any(keyword in question_lower for keyword in logic_keywords):
            return "logical"
        
        # Text analysis
        text_keywords = ['paragraph', 'sentence', 'word', 'meaning', 'define', 'explain', 'describe']
        if any(keyword in question_lower for keyword in text_keywords):
            return "textual"
        
        # Default to general reasoning
        return "general"
    
    def decompose(self, question: str) -> List[str]:
        question_type = self.analyze_question_type(question)
        
        prompt = f"""Analyze this {question_type} question and break it down into 2-4 logical steps. Return ONLY a JSON array of strings.

Question: "{question}"

Examples:
- Math: ["Identify the mathematical operations", "Apply order of operations", "Calculate the result"]
- Explanation: ["Understand the key concepts", "Break down the components", "Provide comprehensive explanation"]
- Logic: ["Analyze the premises", "Apply logical rules", "Derive conclusion"]

JSON:"""
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.llm.generate_response(messages, max_tokens=300)
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                subproblems = json.loads(json_match.group())
                if isinstance(subproblems, list) and len(subproblems) >= 2:
                    return subproblems
        except Exception as e:
            print(f"Decomposition error: {e}")
        
        # Fallback decomposition based on question type
        return self._get_fallback_decomposition(question_type, question)
    
    def _get_fallback_decomposition(self, question_type: str, question: str) -> List[str]:
        """Provide fallback decomposition steps based on question type"""
        fallbacks = {
            "mathematical": [
                f"Analyze the mathematical expression: {question}",
                "Identify the operations and their order",
                "Perform the calculations step by step",
                "Verify the final result"
            ],
            "scientific": [
                f"Understand the scientific concept in: {question}",
                "Break down the key principles involved",
                "Apply relevant scientific laws/theories",
                "Provide the explanation"
            ],
            "historical": [
                f"Identify the historical context of: {question}",
                "Recall relevant historical facts/events",
                "Analyze causes and effects",
                "Provide historical perspective"
            ],
            "textual": [
                f"Analyze the text: {question}",
                "Identify key themes and elements",
                "Interpret meaning and context",
                "Provide comprehensive analysis"
            ],
            "general": [
                f"Understand the question: {question}",
                "Break down into key components",
                "Apply reasoning and knowledge",
                "Formulate complete answer"
            ]
        }
        
        return fallbacks.get(question_type, [
            f"Step 1: Analyze {question}",
            "Step 2: Apply reasoning",
            "Step 3: Provide answer"
        ])

# ============================================
# ENHANCED TOOL SELECTOR
# ============================================
class EnhancedToolSelector:
    def select_tool(self, subproblem: str, question_type: str) -> str:
        text = subproblem.lower()
        
        # Mathematical tools
        if re.search(r'calculate|compute|math|\d+[\+\-\*\/]|result|answer', text):
            return 'calculator'
        elif re.search(r'equation|solve for| x |variable', text):
            return 'equation_solver'
        elif re.search(r'pattern|sequence|next|series', text):
            return 'pattern_recognizer'
        
        # Text analysis tools
        elif re.search(r'analyze text|interpret|meaning|theme', text):
            return 'text_analyzer'
        elif re.search(r'define|definition|what is', text):
            return 'definition_provider'
        elif re.search(r'compare|difference|similar', text):
            return 'comparison_engine'
        
        # Logic and reasoning
        elif re.search(r'logic|reasoning|if then|therefore', text):
            return 'logic_evaluator'
        elif re.search(r'explain|describe|how does|why', text):
            return 'explanation_engine'
        
        # Default to enhanced reasoner
        else:
            return 'enhanced_reasoner'

# ============================================
# ENHANCED EXECUTORS
# ============================================
class EnhancedCalculator:
    def solve(self, text: str) -> str:
        try:
            # Extract mathematical expressions more robustly
            expressions = re.findall(r'[\d+\-*/().^]+', text)
            for expr in expressions:
                try:
                    # Replace ^ with ** for exponentiation
                    clean_expr = expr.replace('^', '**')
                    # Safe evaluation
                    result = eval(clean_expr, {"__builtins__": None}, {
                        "sin": math.sin, "cos": math.cos, "tan": math.tan,
                        "sqrt": math.sqrt, "log": math.log, "exp": math.exp,
                        "pi": math.pi, "e": math.e
                    })
                    return f"The expression {expr} equals {result}"
                except:
                    continue
            
            # Try to extract numbers and operations from text
            numbers = re.findall(r'\d+\.?\d*', text)
            if len(numbers) >= 2:
                if 'add' in text or 'plus' in text or '+' in text:
                    result = sum(float(n) for n in numbers)
                    return f"Sum: {result}"
                elif 'multiply' in text or 'times' in text or 'product' in text or '*' in text:
                    result = 1
                    for n in numbers:
                        result *= float(n)
                    return f"Product: {result}"
                elif 'subtract' in text or 'minus' in text or '-' in text:
                    if len(numbers) == 2:
                        result = float(numbers[0]) - float(numbers[1])
                        return f"Difference: {result}"
                elif 'divide' in text or 'over' in text or '/' in text:
                    if len(numbers) == 2 and float(numbers[1]) != 0:
                        result = float(numbers[0]) / float(numbers[1])
                        return f"Quotient: {result}"
            
            return "Please provide a clearer mathematical expression to calculate."
        except Exception as e:
            return f"Calculation error: {str(e)}"

class EquationSolver:
    def solve(self, text: str) -> str:
        try:
            if not SYMPY_AVAILABLE:
                return "Equation solving requires sympy package. Please install it or use basic calculations."
                
            # Extract equation (look for = sign)
            if '=' in text:
                equation_part = text.split('=')[0] + '=' + text.split('=')[1]
                # Define common variables
                x, y = symbols('x y')
                
                try:
                    # Try to solve using sympy
                    solutions = solve(equation_part, x)
                    if solutions:
                        if len(solutions) == 1:
                            return f"The solution is x = {solutions[0]}"
                        else:
                            return f"Solutions: {', '.join(str(s) for s in solutions)}"
                    else:
                        return "No solution found or the equation is too complex."
                except:
                    return "I can solve linear equations. Please provide an equation like '2x + 5 = 15'."
            else:
                return "Please provide an equation with an '=' sign."
        except Exception as e:
            return f"Equation solving error: {str(e)}"

class TextAnalyzer:
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def analyze(self, text: str) -> str:
        prompt = f"""Analyze this text and provide key insights:

Text: "{text}"

Focus on:
- Main themes and topics
- Key points or arguments
- Overall meaning or purpose
- Any notable patterns or structures

Provide a concise analysis:"""
        
        messages = [{"role": "user", "content": prompt}]
        return self.llm.generate_response(messages, max_tokens=300)

class EnhancedExecutor:
    def __init__(self, llm_client):
        self.llm = llm_client
        self.calculator = EnhancedCalculator()
        self.equation_solver = EquationSolver()
        self.text_analyzer = TextAnalyzer(llm_client)
    
    def execute(self, subproblem: str, tool: str, question_type: str) -> str:
        try:
            if tool == 'calculator':
                return self.calculator.solve(subproblem)
            elif tool == 'equation_solver':
                return self.equation_solver.solve(subproblem)
            elif tool == 'text_analyzer':
                return self.text_analyzer.analyze(subproblem)
            elif tool in ['definition_provider', 'comparison_engine', 'explanation_engine']:
                return self._handle_specialized_tools(subproblem, tool, question_type)
            else:  # enhanced_reasoner and fallback
                return self._handle_general_reasoning(subproblem, question_type)
        except Exception as e:
            return f"Error in execution: {str(e)}"
    
    def _handle_specialized_tools(self, subproblem: str, tool: str, question_type: str) -> str:
        """Handle specialized text analysis tools"""
        prompts = {
            'definition_provider': f"Provide a clear, concise definition for: '{subproblem}'",
            'comparison_engine': f"Compare and contrast the elements in: '{subproblem}'",
            'explanation_engine': f"Explain in detail: '{subproblem}'"
        }
        
        prompt = prompts.get(tool, f"Provide a comprehensive answer to: '{subproblem}'")
        messages = [{"role": "user", "content": prompt}]
        return self.llm.generate_response(messages, max_tokens=350)
    
    def _handle_general_reasoning(self, subproblem: str, question_type: str) -> str:
        """Handle general reasoning with context awareness"""
        context_prompts = {
            "historical": f"Provide historical context and facts for: '{subproblem}'",
            "scientific": f"Provide scientific explanation for: '{subproblem}'",
            "geographical": f"Provide geographical information for: '{subproblem}'",
            "textual": f"Analyze and interpret: '{subproblem}'"
        }
        
        prompt = context_prompts.get(question_type, f"Solve this step: '{subproblem}'")
        messages = [{"role": "user", "content": prompt}]
        return self.llm.generate_response(messages, max_tokens=300)

# ============================================
# ENHANCED AGENTIC SYSTEM
# ============================================
class EnhancedAgenticSystem:
    def __init__(self):
        self.llm = EnhancedGroqClient()
        self.decomposer = EnhancedProblemDecomposer(self.llm)
        self.tool_selector = EnhancedToolSelector()
        self.executor = EnhancedExecutor(self.llm)
    
    def solve_problem(self, question: str) -> Dict[str, Any]:
        try:
            # First, analyze the question type
            question_type = self.decomposer.analyze_question_type(question)
            
            # Decompose based on question type
            subproblems = self.decomposer.decompose(question)
            
            solutions = []
            for subproblem in subproblems:
                tool = self.tool_selector.select_tool(subproblem, question_type)
                solution = self.executor.execute(subproblem, tool, question_type)
                
                solutions.append({
                    'subproblem': subproblem,
                    'tool': tool,
                    'solution': solution
                })
            
            # Generate final answer with context awareness
            final_answer = self._synthesize_answer(solutions, question, question_type)
            
            return {
                'final_answer': final_answer,
                'subproblems': solutions,
                'decomposition_steps': subproblems,
                'question_type': question_type
            }
            
        except Exception as e:
            return self._handle_error(question, str(e))
    
    def _synthesize_answer(self, solutions: List[Dict], question: str, question_type: str) -> str:
        """Synthesize final answer with question type context"""
        if not solutions:
            return "I couldn't generate a solution for this question."
        
        # For single-step solutions, just return the solution
        if len(solutions) == 1:
            return solutions[0]['solution']
        
        # Use LLM to synthesize multi-step answers
        synthesis_prompt = f"""Based on the following step-by-step analysis, provide a comprehensive final answer to the question: "{question}"

Question Type: {question_type}
Steps:
{json.dumps([f"{i+1}. {s['subproblem']} -> {s['solution']}" for i, s in enumerate(solutions)], indent=2)}

Provide a well-structured, comprehensive final answer that directly addresses the original question:"""
        
        messages = [
            {"role": "user", "content": synthesis_prompt}
        ]
        
        return self.llm.generate_response(messages, max_tokens=400)
    
    def _handle_error(self, question: str, error: str) -> Dict[str, Any]:
        """Handle errors gracefully"""
        error_response = {
            'final_answer': f"I encountered an error while processing your question. Please try rephrasing it or ask something else. Error: {error}",
            'subproblems': [{
                'subproblem': 'Error handling',
                'tool': 'error_handler',
                'solution': f'System error: {error}'
            }],
            'decomposition_steps': ['Error occurred during processing'],
            'question_type': 'error'
        }
        return error_response

# ============================================
# API ENDPOINTS
# ============================================
try:
    agent = EnhancedAgenticSystem()
    print("✓ Enhanced Agentic System initialized successfully")
except Exception as e:
    print(f"✗ Failed to initialize Agentic System: {e}")
    agent = None

@app.route('/api/solve', methods=['POST'])
def solve():
    try:
        if agent is None:
            return jsonify({
                'error': 'System not properly initialized. Check GROQ_API_KEY.'
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
                'final_answer': "I'd be happy to help! Please ask me a specific question or problem you'd like me to solve.",
                'subproblems': [],
                'decomposition_steps': [],
                'question_type': 'general'
            })
        
        # Handle common greetings
        greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']
        if question.lower().strip('.!?') in greetings:
            return jsonify({
                'final_answer': "Hello! I'm an Enhanced AI Reasoning Engine. I can help you with mathematical problems, text analysis, explanations, definitions, and much more. What would you like me to help you with?",
                'subproblems': [],
                'decomposition_steps': [],
                'question_type': 'greeting'
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
        'system': 'enhanced-reasoning-engine'
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
        'system': 'Enhanced Reasoning Engine'
    })

@app.route('/api/capabilities', methods=['GET'])
def capabilities():
    return jsonify({
        'capabilities': [
            'Mathematical calculations and equation solving',
            'Text analysis and interpretation',
            'Historical and geographical questions',
            'Scientific explanations',
            'Logical reasoning',
            'Definition and comparison tasks',
            'Step-by-step problem decomposition'
        ],
        'question_types_handled': [
            'mathematical', 'scientific', 'historical', 'geographical',
            'textual', 'logical', 'general'
        ]
    })

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug_mode = os.getenv('DEBUG', 'false').lower() == 'true'
    
    print("🚀 Enhanced AI Reasoning Engine Starting...")
    print(f"📊 Available question types: mathematical, scientific, historical, geographical, textual, logical")
    print(f"🌐 Server will run on: http://localhost:{port}")
    print("✅ System ready to handle complex questions and paragraphs!")
    
    app.run(host='0.0.0.0', port=port, debug=debug_mode)