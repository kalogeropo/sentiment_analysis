from flask import Flask,request, jsonify
import logging



name = "Sentiment analysis API"
logger = logging.getLogger(name)
app = Flask(name)


@app.get("/healthz")
def healthz():
    return jsonify({"ok": True, "name": name})



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)