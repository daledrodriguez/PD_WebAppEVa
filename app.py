from flask import Flask, render_template, request, jsonify
from advanced_rag_main import AdvancedCodeReviewRAG

# Initialize RAG system once in the app context
app = Flask(__name__)

with app.app_context():
    rag_system = AdvancedCodeReviewRAG(debug=True)

# Serve the HTML page
@app.route('/')
def index():
    return render_template('FeedbackPage.html')

# Handle feedback form submission
@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        task_description = request.form.get('ActivityInstruction')
        code_to_review = request.form.get('ActivityCode')

        if not task_description or not code_to_review:
            return jsonify({"error": "Both Activity Instruction and Code Solution are required."}), 400

        review = rag_system.review_code(code_to_review, task_description)

        print(f"Here's My Feedback To Your Code \n {review}")

        return jsonify({"feedback": review}), 200
        
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=False)
