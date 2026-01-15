const tabs = document.querySelectorAll('.toggle__btn');
const panels = document.querySelectorAll('.model');

function activatePanel(id) {
    panels.forEach(panel => {
        panel.classList.toggle('is-hidden', panel.id !== id);
    });
}

tabs.forEach(tab => {
    tab.addEventListener('click', () => {
        if (tab.classList.contains('is-active')) {
            return;
        }
        tabs.forEach(btn => btn.classList.remove('is-active'));
        tab.classList.add('is-active');
        activatePanel(tab.dataset.target);
    });
});

const predictionForm = document.querySelector('#prediction-form');
const resultBox = document.querySelector('.predictor__result');
const resultHeadline = document.querySelector('.predictor__headline');
const resultDetail = document.querySelector('.predictor__detail');
const resultConfidence = document.querySelector('.predictor__confidence');

const BASE_MESSAGES = {
    headline: 'Awaiting sensor input.',
    detail: 'Provide readings to see occupancy probability and recommended action.',
    confidence: ''
};

function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

function normalize(value, center, spread) {
    return clamp((value - center) / spread, -2, 2);
}

function logistic(x) {
    return 1 / (1 + Math.exp(-x));
}

function computeOccupancyProbability({ temperature, humidity, light, voltage }) {
    const tempScore = normalize(temperature, 18.5, 6);
    const humidityScore = normalize(humidity, 38, 18);
    const lightScore = normalize(light, 250, 180);
    const voltageScore = normalize(voltage, 2.75, 0.35);

    const logit = 2.4 * tempScore + 2.1 * humidityScore + 2.8 * lightScore - 1.2 * voltageScore - 0.6;
    return logistic(logit);
}

function interpretProbability(probability) {
    if (probability >= 0.6) {
        return {
            state: 'is-occupied',
            headline: 'Likely Occupied',
            detail: 'Trigger response protocols. Ensemble confidence passes the recall-first threshold tuned during training.',
            confidence: `Probability ≈ ${(probability * 100).toFixed(1)}%`
        };
    }
    if (probability >= 0.45) {
        return {
            state: 'is-occupied',
            headline: 'Borderline Occupancy',
            detail: 'Hold the alert channel open and continue sampling. Conditions resemble noisy positives observed during validation.',
            confidence: `Probability ≈ ${(probability * 100).toFixed(1)}%`
        };
    }
    return {
        state: 'is-clear',
        headline: 'No Occupancy Detected',
        detail: 'Environment resembles idle patterns. Continue monitoring—false negatives remain the greater risk.',
        confidence: `Probability ≈ ${(probability * 100).toFixed(1)}%`
    };
}

function setResultState({ state, headline, detail, confidence }) {
    resultBox.classList.remove('is-occupied', 'is-clear', 'is-error');
    if (state) {
        resultBox.classList.add(state);
    }
    resultHeadline.textContent = headline;
    resultDetail.textContent = detail;
    resultConfidence.textContent = confidence;
}

if (predictionForm && resultBox && resultHeadline && resultDetail && resultConfidence) {
    predictionForm.addEventListener('submit', event => {
        event.preventDefault();

        const temperature = parseFloat(predictionForm.temperature.value);
        const humidity = parseFloat(predictionForm.humidity.value);
        const light = parseFloat(predictionForm.light.value);
        const voltage = parseFloat(predictionForm.voltage.value);

        if ([temperature, humidity, light, voltage].some(value => Number.isNaN(value))) {
            setResultState({
                state: 'is-error',
                headline: 'Input Required',
                detail: 'Please enter numeric readings for every sensor channel before estimating occupancy.',
                confidence: ''
            });
            return;
        }

        const probability = computeOccupancyProbability({ temperature, humidity, light, voltage });
        const interpretation = interpretProbability(probability);
        setResultState(interpretation);
    });

    predictionForm.addEventListener('reset', () => {
        setResultState(BASE_MESSAGES);
    });
}
