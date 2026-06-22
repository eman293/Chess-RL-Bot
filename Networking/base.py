from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/data', methods=['POST'])
def handle_data():
    data = request.json.get('data', '')
    response_message = f"Received: {data}"
    return jsonify({'message': response_message})

if __name__ == '__main__':
    app.run(debug=True)