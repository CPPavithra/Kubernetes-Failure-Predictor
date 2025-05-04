from flask import Flask
from flask_socketio import SocketIO
from threading import Thread
import predictgemini

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Expose socketio instance so other modules can use it
predictgemini.socketio = socketio

@app.route("/")
def index():
    return "K8s Monitoring Dashboard Backend Running"

if __name__ == "__main__":
    # Run the main logic in a background thread
    Thread(target=predictgemini.main).start()
    socketio.run(app, host="0.0.0.0", port=5000)

