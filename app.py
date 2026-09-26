from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

import numpy as np
import os
import uuid


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

# Validator threshold
# Class 0 = lung_ct
# Class 1 = not_lung_ct
VALIDATOR_THRESHOLD = 0.50


# =========================================================
# Load Models
# =========================================================

print("Loading LungScopeAI models...")

model = tf.keras.models.load_model(
    CLASSIFIER_MODEL_PATH
)

validator_model = tf.keras.models.load_model(
    VALIDATOR_MODEL_PATH
)

print("Classification model loaded.")
print("Lung CT validator loaded.")


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

    return render_template(
        "index.html"
    )


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
    # Get patient information
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
    # Validate image extension
    # -----------------------------------------------------

    allowed_extensions = {
        "png",
        "jpg",
        "jpeg"
    }

    original_filename = secure_filename(
        file.filename
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
    # Create unique filename
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
    # Save image
    # -----------------------------------------------------

    file.save(filepath)


    # -----------------------------------------------------
    # Load image
    # -----------------------------------------------------

    try:

        img = image.load_img(
            filepath,
            target_size=(IMG_SIZE, IMG_SIZE)
        )

    except Exception:

        return render_template(
            "index.html",
            error="The uploaded file could not be processed as an image."
        )


    # -----------------------------------------------------
    # Convert image to array
    # -----------------------------------------------------

    img_array = image.img_to_array(
        img
    )


    # -----------------------------------------------------
    # Add batch dimension
    # -----------------------------------------------------

    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    # -----------------------------------------------------
    # EfficientNet preprocessing
    # -----------------------------------------------------

    img_array = preprocess_input(
        img_array
    )


    # =====================================================
    # STEP 1 — LUNG CT IMAGE VALIDATION
    # =====================================================

    validator_prediction = validator_model.predict(
        img_array,
        verbose=0
    )

    # The validator was trained with:
    #
    # class 0 = lung_ct
    # class 1 = not_lung_ct
    #
    # Therefore sigmoid output represents:
    # probability/score of NOT being a lung CT.

    not_lung_ct_score = float(
        validator_prediction[0][0]
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
    # STEP 2 — 4-CLASS LUNG IMAGE CLASSIFICATION
    # =====================================================

    predictions = model.predict(
        img_array,
        verbose=0
    )


    # -----------------------------------------------------
    # Get predicted class
    # -----------------------------------------------------

    index = int(
        np.argmax(
            predictions[0]
        )
    )


    # -----------------------------------------------------
    # Confidence
    # -----------------------------------------------------

    confidence = float(
        predictions[0][index] * 100
    )


    # -----------------------------------------------------
    # Prediction name
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
    # STEP 3 — RENDER RESULT
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
# Run Application
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )