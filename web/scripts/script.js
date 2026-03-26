const predictionForm = document.querySelector('#prediction-form');
const resultBox = document.querySelector('.predictor__result');
const resultHeadline = document.querySelector('.predictor__headline');
const resultDetail = document.querySelector('.predictor__detail');
const resultConfidence = document.querySelector('.predictor__confidence');
const quickValueButtons = document.querySelectorAll('.quick-values__btn');
const modelSelect = document.querySelector('#model-select');
const activeModelLabel = document.querySelector('#active-model-label');

const BASE_MESSAGES = {
    headline: 'Awaiting sensor input.',
    detail: 'Enter current readings and run detection.',
    confidence: ''
};

function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

function normalizeRange(value, min, max) {
    return clamp((value - min) / (max - min), 0, 1);
}

function logistic(x) {
    return 1 / (1 + Math.exp(-x));
}

function computeRandomForestProbability({ temperature, humidity, light, voltage }) {
    const tempGate = temperature > 18 ? 1 : 0;
    const humidityGate = humidity > 35 ? 1 : 0;
    const lightGate = light > 250 ? 1 : 0;

    const tempSignal = normalizeRange(temperature, 15, 30);
    const humiditySignal = normalizeRange(humidity, 30, 65);
    const lightSignal = normalizeRange(light, 100, 650);
    const voltageSignal = 1 - Math.abs(clamp(voltage, 2.2, 3.4) - 2.75) / 0.65;

    // Weighting mirrors the notebook's occupancy definition where light/humidity/temperature drive labels.
    const logit =
        -2.2 +
        1.25 * tempGate +
        1.35 * humidityGate +
        1.7 * lightGate +
        0.85 * tempSignal +
        0.9 * humiditySignal +
        1.15 * lightSignal +
        0.25 * voltageSignal +
        0.55 * tempGate * lightGate;

    return logistic(logit);
}

function computeKMeansProbability({ temperature, humidity, light, voltage }) {
    const tempSignal = normalizeRange(temperature, 12, 34);
    const humiditySignal = normalizeRange(humidity, 20, 70);
    const lightSignal = normalizeRange(light, 0, 900);
    const voltageSignal = normalizeRange(voltage, 2.2, 3.3);

    const occupiedCentroid = { temp: 0.62, humidity: 0.58, light: 0.66, voltage: 0.5 };
    const clearCentroid = { temp: 0.38, humidity: 0.36, light: 0.22, voltage: 0.5 };

    const dOcc =
        (tempSignal - occupiedCentroid.temp) ** 2 +
        (humiditySignal - occupiedCentroid.humidity) ** 2 +
        (lightSignal - occupiedCentroid.light) ** 2 +
        (voltageSignal - occupiedCentroid.voltage) ** 2;

    const dClear =
        (tempSignal - clearCentroid.temp) ** 2 +
        (humiditySignal - clearCentroid.humidity) ** 2 +
        (lightSignal - clearCentroid.light) ** 2 +
        (voltageSignal - clearCentroid.voltage) ** 2;

    return logistic((dClear - dOcc) * 3.2);
}

function getModelName(modelKey) {
    return modelKey === 'kmeans' ? 'K-Means' : 'Random Forest';
}

function computeProbabilityByModel(sensorValues, modelKey) {
    if (modelKey === 'kmeans') {
        return computeKMeansProbability(sensorValues);
    }
    return computeRandomForestProbability(sensorValues);
}

function interpretProbability(probability, modelKey) {
    const modelName = getModelName(modelKey);
    if (probability >= 0.5) {
        return {
            state: 'is-occupied',
            headline: 'Human Presence Detected',
            detail: `${modelName} confidence is above decision threshold. Keep monitoring or trigger the next workflow.`,
            confidence: `${modelName} presence probability: ${(probability * 100).toFixed(1)}%`
        };
    }
    if (probability >= 0.4) {
        return {
            state: 'is-occupied',
            headline: 'Possible Human Presence',
            detail: `${modelName} signal is borderline. Capture another reading for confirmation.`,
            confidence: `${modelName} presence probability: ${(probability * 100).toFixed(1)}%`
        };
    }
    return {
        state: 'is-clear',
        headline: 'No Human Presence Detected',
        detail: `${modelName} output is below the alert threshold.`,
        confidence: `${modelName} presence probability: ${(probability * 100).toFixed(1)}%`
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
    if (modelSelect && activeModelLabel) {
        activeModelLabel.textContent = getModelName(modelSelect.value);
        modelSelect.addEventListener('change', () => {
            activeModelLabel.textContent = getModelName(modelSelect.value);
            setResultState({
                state: '',
                headline: 'Model Changed',
                detail: `Now using ${getModelName(modelSelect.value)} for detection.`,
                confidence: ''
            });
        });
    }

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
                detail: 'Please enter numeric values for all sensor fields before running detection.',
                confidence: ''
            });
            return;
        }

        if (humidity < 0 || humidity > 100 || light < 0 || voltage < 0) {
            setResultState({
                state: 'is-error',
                headline: 'Invalid Sensor Range',
                detail: 'Humidity must be 0-100%, while light and voltage cannot be negative.',
                confidence: ''
            });
            return;
        }

        const selectedModel = modelSelect ? modelSelect.value : 'rf';
        const probability = computeProbabilityByModel({ temperature, humidity, light, voltage }, selectedModel);
        const interpretation = interpretProbability(probability, selectedModel);
        setResultState(interpretation);
    });

    predictionForm.addEventListener('reset', () => {
        setResultState(BASE_MESSAGES);
    });

    quickValueButtons.forEach(button => {
        button.addEventListener('click', () => {
            predictionForm.temperature.value = button.dataset.temp;
            predictionForm.humidity.value = button.dataset.humidity;
            predictionForm.light.value = button.dataset.light;
            predictionForm.voltage.value = button.dataset.voltage;
            setResultState({
                state: '',
                headline: 'Sample Loaded',
                detail: 'Press Run Detection to evaluate these values.',
                confidence: ''
            });
        });
    });
}
