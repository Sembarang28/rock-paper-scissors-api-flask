from tensorflow import keras
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify, make_response
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

model = keras.models.load_model("model.h5")

@app.route("/upload", methods=["POST"])
def upload():
    apikey = request.headers.get('apikey')

    if "image" not in request.files:
        return jsonify({"error": "Missing required request"})

    image_file = request.files["image"]
    image_file.save("img.jpg")
    path = "img.jpg"

    # if classes[0][0]==1:
    #   print('paper')
    # elif classes[0][1]==1:
    #   print('rock')
    # elif classes[0][2]==1:
    #   print('scissors')
    # else:
    #   print('unknown')

    try:
        img = image.load_img(path, target_size=(100,150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        print(path)
        classes = model.predict(images, batch_size=20)
        print("Predicted class:", classes[0])
        # Add the following line to print the perspective class
        perspective_class = np.argmax(classes[0])
        print("Perspective class:", perspective_class)
        
        if classes[0][0]==1:
          name = 'paper'
        elif classes[0][1]==1:
          name = 'rock'
        elif classes[0][2]==1:
          name = 'scissors'
        else:
          name = 'unknown'
      
        os.remove(path)
        return jsonify({"predicted_class": str(perspective_class), "name": name})

    except Exception as e:
        return jsonify({"error": str(e)})
      
@app.route("/test", methods=["GET"])
def test():
  return jsonify({ "status": True, "message": "Hello, World!!"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))