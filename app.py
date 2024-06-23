from flask import Flask, request, jsonify, send_from_directory
import subprocess

app = Flask(__name__, static_folder='.')

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/process', methods=['POST'])
def process_text():
    data = request.get_json()
    input_text = data['text']
    
    # Записываем текст в input.txt
    with open('input.txt', 'w') as f:
        f.write(input_text)
    
    # Запускаем main.py
    subprocess.run(['python', 'main.py'])
    
    # Читаем результат из output.txt
    with open('output.txt', 'r') as f:
        output_text = f.read()
    
    return jsonify({'output': output_text})

if __name__ == '__main__':
    app.run(debug=True)
