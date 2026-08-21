const reviewInput = document.getElementById("review");
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


reviewInput.addEventListener("input", () => {

    charCount.textContent = reviewInput.value.length;

});


analyzeBtn.addEventListener("click", async () => {

    const text = reviewInput.value.trim();

    if (!text) {
        showError("Please enter a customer review.");
        return;
    }

    result.classList.add("hidden");
    errorBox.classList.add("hidden");
    loading.classList.remove("hidden");

    try {

        const response = await fetch(
            "http://127.0.0.1:8000/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    text: text
                })
            }
        );


        const data = await response.json();


        if (!response.ok) {
            throw new Error(
                data.detail || "Prediction failed."
            );
        }


        sentiment.textContent = data.sentiment;


        const confidence =
            data.confidence * 100;

        confidenceValue.textContent =
            confidence.toFixed(2) + "%";

        confidenceBar.style.width =
            confidence + "%";


        negativeProbability.textContent =
            (data.probabilities.negative * 100).toFixed(2) + "%";

        neutralProbability.textContent =
            (data.probabilities.neutral * 100).toFixed(2) + "%";

        positiveProbability.textContent =
            (data.probabilities.positive * 100).toFixed(2) + "%";


        result.classList.remove("hidden");

    }

    catch (error) {

        showError(
            "Could not connect to the sentiment API."
        );

        console.error(error);

    }

    finally {

        loading.classList.add("hidden");

    }

});


function showError(message) {

    errorBox.textContent = message;

    errorBox.classList.remove("hidden");

}