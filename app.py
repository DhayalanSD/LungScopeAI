# =========================================================
# LungScopeAI - Flask Production Application
# =========================================================

import os

# =========================================================
# TensorFlow Environment Configuration
# IMPORTANT:
# These must be set BEFORE importing TensorFlow.
# =========================================================

# Force CPU-only inference on Render
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Reduce TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Limit OpenMP threads
os.environ["OMP_NUM_THREADS"] = "1"


# =========================================================
# Imports
# =========================================================

import gc
import uuid

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import numpy as np

import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input


# =========================================================
# TensorFlow CPU Configuration
# =========================================================

try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass


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

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Maximum upload size = 10 MB
app.config["MAX_CONTENT_LENGTH"] = (
    10 * 1024 * 1024
)


# =========================================================
# Model Configuration
# =========================================================

# IMPORTANT:
# Use your BEST EfficientNetB1 model.
CLASSIFIER_MODEL_PATH = (
    "models/best_b1_model.keras"
)

# Lung CT validator
VALIDATOR_MODEL_PATH = (
    "models/lung_ct_validator.keras"
)

# Model input size
IMG_SIZE = 224

# Validator threshold
VALIDATOR_THRESHOLD = 0.50


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
# TensorFlow Memory Cleanup
# =========================================================

def clear_tensorflow_memory():

    gc.collect()

    try:
        tf.keras.backend.clear_session()
    except Exception:
        pass

    gc.collect()


# =========================================================
# Prepare Image
# =========================================================

def prepare_image(filepath):

    img = image.load_img(
        filepath,
        target_size=(
            IMG_SIZE,
            IMG_SIZE
        )
    )

    img_array = image.img_to_array(
        img
    )

    # Release PIL image
    del img

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = preprocess_input(
        img_array
    )

    # Ensure float32
    img_array = img_array.astype(
        np.float32,
        copy=False
    )

    return img_array


# =========================================================
# Load Validator
# =========================================================

def load_validator():

    print(
        "Loading Lung CT validator..."
    )

    validator = tf.keras.models.load_model(
        VALIDATOR_MODEL_PATH,
        compile=False
    )

    print(
        "Lung CT validator loaded."
    )

    return validator


# =========================================================
# Load EfficientNetB1
# =========================================================

def load_classifier():

    print(
        "Loading EfficientNetB1 classifier..."
    )

    classifier = tf.keras.models.load_model(
        CLASSIFIER_MODEL_PATH,
        compile=False
    )

    print(
        "EfficientNetB1 classifier loaded."
    )

    return classifier


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

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # =====================================================
    # CHECK IMAGE
    # =====================================================

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


    # =====================================================
    # PATIENT INFORMATION
    # =====================================================

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


    # =====================================================
    # FILE VALIDATION
    # =====================================================

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
            error=(
                "Only PNG, JPG and JPEG "
                "images are allowed."
            )
        )


    # =====================================================
    # UNIQUE FILENAME
    # =====================================================

    unique_filename = (
        uuid.uuid4().hex
        + "."
        + extension
    )

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        unique_filename
    )


    # =====================================================
    # SAVE IMAGE
    # =====================================================

    try:

        file.save(
            filepath
        )

    except Exception as e:

        print(
            "File save error:",
            repr(e)
        )

        return render_template(
            "index.html",
            error=(
                "The uploaded image "
                "could not be saved."
            )
        )


    # =====================================================
    # PREPARE IMAGE
    # =====================================================

    img_array = None

    try:

        img_array = prepare_image(
            filepath
        )

    except Exception as e:

        print(
            "Image processing error:",
            repr(e)
        )

        try:

            if os.path.exists(filepath):

                os.remove(filepath)

        except Exception:
            pass

        return render_template(
            "index.html",
            error=(
                "The uploaded file could "
                "not be processed as an image."
            )
        )


    # =====================================================
    # STEP 1
    # LUNG CT VALIDATION
    # =====================================================

    validator = None
    validator_output = None

    try:

        print(
            "========================================"
        )

        print(
            "STEP 1: Lung CT validation"
        )

        validator = load_validator()


        # -------------------------------------------------
        # Direct TensorFlow inference
        # -------------------------------------------------

        validator_output = validator(
            img_array,
            training=False
        )


        # -------------------------------------------------
        # Convert output to NumPy
        # -------------------------------------------------

        validator_values = (
            validator_output.numpy()
            .reshape(-1)
        )


        not_lung_ct_score = float(
            validator_values[0]
        )


        print(
            "Validator score:",
            not_lung_ct_score
        )


    except Exception as e:

        print(
            "Validator error:",
            repr(e)
        )

        # Cleanup
        try:
            del validator
        except Exception:
            pass

        try:
            del validator_output
        except Exception:
            pass

        try:
            del validator_values
        except Exception:
            pass

        try:
            del img_array
        except Exception:
            pass

        clear_tensorflow_memory()

        return render_template(
            "index.html",
            error=(
                "The image validation process "
                "failed. Please try again."
            )
        )


    # =====================================================
    # DELETE VALIDATOR
    # BEFORE LOADING B1
    # =====================================================

    try:
        del validator
    except Exception:
        pass

    try:
        del validator_output
    except Exception:
        pass

    try:
        del validator_values
    except Exception:
        pass


    # IMPORTANT:
    # Clear TensorFlow/Keras state before
    # loading EfficientNetB1.
    clear_tensorflow_memory()


    # =====================================================
    # STEP 1 RESULT
    # UNSUPPORTED IMAGE
    # =====================================================

    if (
        not_lung_ct_score
        >= VALIDATOR_THRESHOLD
    ):

        print(
            "Result: Unsupported image"
        )

        print(
            "Validator percentage:",
            round(
                not_lung_ct_score * 100,
                2
            )
        )


        image_path = (
            "/static/uploads/"
            + unique_filename
        )


        # Free image tensor
        try:
            del img_array
        except Exception:
            pass

        clear_tensorflow_memory()


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
    # STEP 2
    # FOUR-CLASS CLASSIFICATION
    # =====================================================

    classifier = None
    classifier_output = None
    prediction_values = None

    try:

        print(
            "========================================"
        )

        print(
            "STEP 2: EfficientNetB1 classification"
        )

        classifier = load_classifier()


        # -------------------------------------------------
        # Direct TensorFlow inference
        # -------------------------------------------------

        classifier_output = classifier(
            img_array,
            training=False
        )


        # -------------------------------------------------
        # Convert to NumPy
        # -------------------------------------------------

        prediction_values = (
            classifier_output.numpy()
            .reshape(-1)
        )


        # -------------------------------------------------
        # Find class
        # -------------------------------------------------

        index = int(
            np.argmax(
                prediction_values
            )
        )


        # -------------------------------------------------
        # Confidence
        # -------------------------------------------------

        confidence = float(
            prediction_values[index] * 100
        )


        # -------------------------------------------------
        # Class name
        # -------------------------------------------------

        prediction_name = classes[index]


        print(
            "Prediction:",
            prediction_name
        )

        print(
            "Confidence:",
            round(
                confidence,
                2
            )
        )


    except Exception as e:

        print(
            "Classification error:",
            repr(e)
        )


        # Cleanup
        try:
            del classifier
        except Exception:
            pass

        try:
            del classifier_output
        except Exception:
            pass

        try:
            del prediction_values
        except Exception:
            pass

        try:
            del img_array
        except Exception:
            pass

        clear_tensorflow_memory()


        return render_template(
            "index.html",
            error=(
                "The classification process "
                "failed. Please try again."
            )
        )


    # =====================================================
    # RESULT IMAGE PATH
    # =====================================================

    image_path = (
        "/static/uploads/"
        + unique_filename
    )


    # =====================================================
    # CLEANUP CLASSIFIER
    # =====================================================

    try:
        del classifier
    except Exception:
        pass

    try:
        del classifier_output
    except Exception:
        pass

    try:
        del prediction_values
    except Exception:
        pass

    try:
        del img_array
    except Exception:
        pass

    clear_tensorflow_memory()


    # =====================================================
    # RETURN RESULT
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
# File Too Large
# =========================================================

@app.errorhandler(413)
def request_entity_too_large(error):

    return render_template(
        "index.html",

        error=(
            "Image is too large. "
            "Please upload an image "
            "smaller than 10 MB."
        )

    ), 413


# =========================================================
# Health Check
# =========================================================

@app.route("/health")
def health():

    return {
        "status": "ok",
        "service": "LungScopeAI"
    }, 200


# =========================================================
# Start Application
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",

        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        ),

        debug=False
    )