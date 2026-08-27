const reviewInput = document.getElementById("review");
const modelSelect = document.getElementById("model");
const charCount = document.getElementById("charCount");
const analyzeBtn = document.getElementById("analyzeBtn");

const loading = document.getElementById("loading");
const result = document.getElementById("result");
const errorBox = document.getElementById("error");

const sentiment = document.getElementById("sentiment");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceBar = document.getElementById("confidenceBar");

const negativeProbability =
document.getElementById("negativeProbability");

const neutralProbability =
document.getElementById("neutralProbability");

const positiveProbability =
document.getElementById("positiveProbability");

// ==============================
// CHARACTER COUNT
// ==============================

reviewInput.addEventListener("input", () => {


charCount.textContent =
    reviewInput.value.length;


});

// ==============================
// ANALYZE SENTIMENT
// ==============================

analyzeBtn.addEventListener("click", async () => {


const text = reviewInput.value.trim();

const model = modelSelect.value;


// --------------------------
// Validate input
// --------------------------

if (!text) {

    showError(
        "Please enter a customer review."
    );

    return;
}


// --------------------------
// Reset UI
// --------------------------

result.classList.add("hidden");

errorBox.classList.add("hidden");

loading.classList.remove("hidden");

analyzeBtn.disabled = true;


try {

    // --------------------------
    // Send request to API
    // --------------------------

    const response = await fetch(
        "http://127.0.0.1:8000/predict",
        {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                text: text,
                model: model
            })
        }
    );


    // --------------------------
    // Read response
    // --------------------------

    const data = await response.json();


    // --------------------------
    // Handle API error
    // --------------------------

    if (!response.ok) {

        throw new Error(
            data.detail || "Prediction failed."
        );
    }


    // --------------------------
    // Display sentiment
    // --------------------------

    sentiment.textContent =
        data.sentiment;


    // --------------------------
    // Confidence
    // --------------------------

    const confidence =
        data.confidence * 100;

    confidenceValue.textContent =
        confidence.toFixed(2) + "%";

    confidenceBar.style.width =
        confidence + "%";


    // --------------------------
    // Probabilities
    // --------------------------

    negativeProbability.textContent =
        getProbability(
            data.probabilities,
            "negative"
        );

    neutralProbability.textContent =
        getProbability(
            data.probabilities,
            "neutral"
        );

    positiveProbability.textContent =
        getProbability(
            data.probabilities,
            "positive"
        );


    // --------------------------
    // Show result
    // --------------------------

    result.classList.remove("hidden");

}


catch (error) {

    console.error(error);

    showError(
        error.message ||
        "Could not connect to the sentiment API."
    );

}


finally {

    loading.classList.add("hidden");

    analyzeBtn.disabled = false;

}


});

// ==============================
// GET PROBABILITY
// ==============================

function getProbability(
probabilities,
label
) {


// Handles both lowercase and
// uppercase model labels.

const key = Object.keys(
    probabilities
).find(
    key =>
        key.toLowerCase() ===
        label.toLowerCase()
);


if (!key) {
    return "0%";
}


return (
    probabilities[key] * 100
).toFixed(2) + "%";


}

// ==============================
// SHOW ERROR
// ==============================

function showError(message) {


errorBox.textContent =
    message;

errorBox.classList.remove(
    "hidden"
);


}
