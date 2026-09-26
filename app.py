import os

# Reduce TensorFlow logging before importing TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

import numpy as np
import uuid


# =========================================================
# TensorFlow CPU Optimization
# =========================================================

# LungScopeAI only performs inference.
# Limiting TensorFlow threads helps prevent excessive
# CPU and memory usage on small cloud instances.

try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except RuntimeError:
    pass


# =========================================================
# Flask App
# =========================================================

app = Flask(__name__)


# =========================================================
# Upload Configuration
# =========================================================

UPLOAD_FOLDER = "static/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# =========================================================
# Model Configuration
# =========================================================

CLASSIFIER_MODEL_PATH = "models/best_model.keras"
VALIDATOR_MODEL_PATH = "models/lung_ct_validator.keras"

IMG_SIZE = 224

VALIDATOR_THRESHOLD = 0.50


# =========================================================
# Load Models
# =========================================================

print("Loading LungScopeAI models...")

# compile=False because these models are used only for inference.
validator_model = tf.keras.models.load_model(
    VALIDATOR_MODEL_PATH,
    compile=False
)

print("Lung CT validator loaded.")

model = tf.keras.models.load_model(
    CLASSIFIER_MODEL_PATH,
    compile=False
)

print("Classification model loaded.")


# =========================================================
# Classification Classes
# =========================================================

classes = [
    "adenocarcinoma",
    "large.cell.carcinoma",
    "normal",
    "squamous.cell.carcinoma"
]


# =========================================================
# Home Page
# =========================================================

@app.route("/")
def home():

    return render_template("index.html")


# =========================================================
# Prediction
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():

    # -----------------------------------------------------
    # Check image
    # -----------------------------------------------------

    if "image" not in request.files:

        return render_template(
            "index.html",
            error="No image was uploaded."
        )

    file = request.files["image"]

    if file.filename == "":

        return render_template(
            "index.html",
            error="Please select an image."
        )


    # -----------------------------------------------------
    # Patient Information
    # -----------------------------------------------------

    patient_name = request.form.get(
        "patient_name",
        ""
    ).strip()

    patient_age = request.form.get(
        "patient_age",
        ""
    ).strip()

    patient_place = request.form.get(
        "patient_place",
        ""
    ).strip()


    # -----------------------------------------------------
    # Validate Extension
    # -----------------------------------------------------

    allowed_extensions = {
        "png",
        "jpg",
        "jpeg"
    }

    original_filename = secure_filename(
        file.filename
    )

    if "." not in original_filename:

        return render_template(
            "index.html",
            error="Please upload a valid image file."
        )

    extension = (
        original_filename
        .rsplit(".", 1)[-1]
        .lower()
    )

    if extension not in allowed_extensions:

        return render_template(
            "index.html",
            error="Only PNG, JPG and JPEG images are allowed."
        )


    # -----------------------------------------------------
    # Unique Filename
    # -----------------------------------------------------

    unique_filename = (
        uuid.uuid4().hex
        + "."
        + extension
    )

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        unique_filename
    )


    # -----------------------------------------------------
    # Save Image
    # -----------------------------------------------------

    file.save(filepath)


    # -----------------------------------------------------
    # Load Image
    # -----------------------------------------------------

    try:

        img = image.load_img(
            filepath,
            target_size=(IMG_SIZE, IMG_SIZE)
        )

        img_array = image.img_to_array(img)

    except Exception:

        return render_template(
            "index.html",
            error="The uploaded file could not be processed as an image."
        )


    # -----------------------------------------------------
    # Prepare Tensor
    # -----------------------------------------------------

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = preprocess_input(
        img_array
    )

    # Convert once to float32
    img_array = img_array.astype(
        np.float32,
        copy=False
    )


    # =====================================================
    # STEP 1 — LUNG CT VALIDATION
    # =====================================================

    # Direct model call is lighter than model.predict()
    validator_output = validator_model(
        img_array,
        training=False
    )

    not_lung_ct_score = float(
        validator_output.numpy().reshape(-1)[0]
    )


    # -----------------------------------------------------
    # Unsupported Image
    # -----------------------------------------------------

    if not_lung_ct_score >= VALIDATOR_THRESHOLD:

        image_path = (
            "/static/uploads/"
            + unique_filename
        )

        return render_template(
            "index.html",

            unsupported_image=True,

            validator_score=round(
                not_lung_ct_score * 100,
                2
            ),

            image_path=image_path,

            patient_name=patient_name,

            patient_age=patient_age,

            patient_place=patient_place
        )


    # =====================================================
    # STEP 2 — 4-CLASS CLASSIFICATION
    # =====================================================

    classifier_output = model(
        img_array,
        training=False
    )

    prediction_values = (
        classifier_output.numpy().reshape(-1)
    )


    # -----------------------------------------------------
    # Get Predicted Class
    # -----------------------------------------------------

    index = int(
        np.argmax(prediction_values)
    )


    # -----------------------------------------------------
    # Confidence
    # -----------------------------------------------------

    confidence = float(
        prediction_values[index] * 100
    )


    # -----------------------------------------------------
    # Prediction Name
    # -----------------------------------------------------

    prediction_name = classes[index]


    # -----------------------------------------------------
    # Image URL
    # -----------------------------------------------------

    image_path = (
        "/static/uploads/"
        + unique_filename
    )


    # =====================================================
    # STEP 3 — RESULT
    # =====================================================

    return render_template(
        "index.html",

        prediction=prediction_name,

        confidence=round(
            confidence,
            2
        ),

        image_path=image_path,

        patient_name=patient_name,

        patient_age=patient_age,

        patient_place=patient_place,

        unsupported_image=False
    )


# =========================================================
# Local Development
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )