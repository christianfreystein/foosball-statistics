<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tischfußball Heatmap App</title>
    <!-- Tailwind CSS CDN für schnelles Styling -->
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f3f4f6;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .container {
            width: 100%;
            max-width: 800px;
            background-color: white;
            padding: 2.5rem;
            border-radius: 1.5rem;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        }
        canvas {
            border: 2px solid #e5e7eb;
            border-radius: 0.75rem;
            background-color: #f9fafb;
            cursor: crosshair;
            display: block;
            width: 100%;
            height: auto;
        }
        .dot {
            position: absolute;
            background-color: #ef4444;
            border: 2px solid white;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            transform: translate(-50%, -50%);
        }
        .heatmap-canvas {
            background-color: black; /* Hintergrund für die Heatmap */
            border: 2px solid #e5e7eb;
            border-radius: 0.75rem;
        }
    </style>
</head>
<body>

<div class="container">
    <h1 class="text-3xl md:text-4xl font-bold text-center text-gray-800 mb-8">Tischfußball Heatmap</h1>

    <!-- Hauptansichten -->
    <div id="view-input" class="view">
        <p class="text-center text-gray-600 mb-6">Bitte geben Sie die RTMP-Stream-URL ein, um fortzufahren.</p>
        <div class="mb-4">
            <label for="rtmpUrl" class="block text-gray-700 font-medium mb-2">RTMP URL</label>
            <input type="text" id="rtmpUrl" value="rtmp://localhost/live/teststream" class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 transition duration-200">
        </div>
        <button id="nextBtn" class="w-full bg-blue-600 text-white font-semibold py-3 px-6 rounded-lg shadow-lg hover:bg-blue-700 transition duration-300 transform hover:scale-105">
            Weiter zur Konfiguration
        </button>
    </div>

    <div id="view-config" class="view hidden">
        <p class="text-center text-gray-600 mb-6">Klicken Sie auf das Bild, um die vier Ecken des Spielfelds in dieser Reihenfolge zu markieren: oben links, oben rechts, unten rechts, unten links.</p>
        <div class="relative mb-6">
            <img id="streamScreenshot" class="w-full h-auto rounded-lg shadow-md" alt="Stream Screenshot" src="https://placehold.co/700x400/1e293b/d1d5db?text=Screenshot+vom+Livestream">
            <canvas id="screenshotCanvas" class="absolute inset-0 w-full h-full"></canvas>
            <div id="corner-dots"></div>
        </div>

        <div class="flex space-x-4 mb-4">
             <button id="resetCornersBtn" class="flex-1 bg-gray-400 text-white font-semibold py-3 rounded-lg shadow-md hover:bg-gray-500 transition duration-300">
                Ecken zurücksetzen
            </button>
            <button id="confirmAreaBtn" class="flex-1 bg-green-600 text-white font-semibold py-3 rounded-lg shadow-lg hover:bg-green-700 transition duration-300 transform hover:scale-105" disabled>
                Bereich bestätigen
            </button>
        </div>
         <p id="cornerMessage" class="text-center text-sm text-red-500"></p>
    </div>

    <div id="view-processing" class="view hidden">
        <div class="text-center mb-6">
            <h2 class="text-2xl font-semibold text-gray-800 mb-2">Datenverarbeitung</h2>
            <p id="statusMessage" class="text-gray-600">Bereit zum Starten der Datenerfassung.</p>
        </div>
        <div class="flex space-x-4">
            <button id="startBtn" class="flex-1 bg-blue-600 text-white font-semibold py-3 px-6 rounded-lg shadow-lg hover:bg-blue-700 transition duration-300 transform hover:scale-105">
                Start
            </button>
            <button id="stopBtn" class="flex-1 bg-red-600 text-white font-semibold py-3 px-6 rounded-lg shadow-lg hover:bg-red-700 transition duration-300 transform hover:scale-105" disabled>
                Stop
            </button>
        </div>
    </div>

    <div id="view-heatmap" class="view hidden">
        <h2 class="text-2xl font-semibold text-center text-gray-800 mb-4">Ball-Heatmap</h2>
        <canvas id="heatmapCanvas" class="heatmap-canvas w-full h-[400px] mb-6"></canvas>
        <button id="restartBtn" class="w-full bg-blue-600 text-white font-semibold py-3 px-6 rounded-lg shadow-lg hover:bg-blue-700 transition duration-300 transform hover:scale-105">
            Neu starten
        </button>
    </div>

    <!-- Modaler Dialog für Fehlermeldungen -->
    <div id="modal" class="fixed inset-0 bg-gray-900 bg-opacity-50 flex items-center justify-center p-4 hidden">
        <div class="bg-white p-8 rounded-lg shadow-xl max-w-sm w-full">
            <h3 class="text-xl font-bold text-gray-800 mb-4">Achtung</h3>
            <p id="modal-message" class="text-gray-700 mb-6"></p>
            <button id="modal-close-btn" class="w-full bg-blue-600 text-white font-semibold py-3 rounded-lg hover:bg-blue-700 transition duration-300">Schließen</button>
        </div>
    </div>
</div>

<script>
    // Ansichten und Elemente
    const views = {
        input: document.getElementById('view-input'),
        config: document.getElementById('view-config'),
        processing: document.getElementById('view-processing'),
        heatmap: document.getElementById('view-heatmap')
    };

    // UI-Elemente
    const rtmpUrlInput = document.getElementById('rtmpUrl');
    const nextBtn = document.getElementById('nextBtn');
    const screenshotImage = document.getElementById('streamScreenshot');
    const screenshotCanvas = document.getElementById('screenshotCanvas');
    const confirmAreaBtn = document.getElementById('confirmAreaBtn');
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const statusMessage = document.getElementById('statusMessage');
    const heatmapCanvas = document.getElementById('heatmapCanvas');
    const restartBtn = document.getElementById('restartBtn');
    const cornerMessage = document.getElementById('cornerMessage');
    const resetCornersBtn = document.getElementById('resetCornersBtn');
    const modal = document.getElementById('modal');
    const modalMessage = document.getElementById('modal-message');
    const modalCloseBtn = document.getElementById('modal-close-btn');

    // App-Zustand
    let currentView = 'input';
    let corners = [];
    let processingInterval = null;

    // Funktionen für die UI-Steuerung
    function setView(viewName) {
        Object.values(views).forEach(view => view.classList.add('hidden'));
        views[viewName].classList.remove('hidden');
        currentView = viewName;
    }

    function showModal(message) {
        modalMessage.textContent = message;
        modal.classList.remove('hidden');
    }

    // Event-Handler

    // Initialisiere die Canvas-Größe basierend auf dem Bild
    screenshotImage.onload = () => {
        screenshotCanvas.width = screenshotImage.width;
        screenshotCanvas.height = screenshotImage.height;
    };

    nextBtn.addEventListener('click', () => {
        // Im realen Szenario: Senden der RTMP-URL an den Backend-Server
        // und Warten auf einen Screenshot.
        // const rtmpUrl = rtmpUrlInput.value;
        // fetch('/api/screenshot', { method: 'POST', body: JSON.stringify({ rtmpUrl }) })
        //     .then(response => response.json())
        //     .then(data => {
        //         screenshotImage.src = data.screenshotUrl;
        //         setView('config');
        //     });

        // Simulation des Backend-Aufrufs:
        setView('config');
    });

    screenshotCanvas.addEventListener('click', (event) => {
        if (corners.length >= 4) {
            showModal("Sie haben bereits vier Ecken markiert. Setzen Sie die Ecken zurück, um neue zu wählen.");
            return;
        }

        const rect = screenshotCanvas.getBoundingClientRect();
        const x = (event.clientX - rect.left) / rect.width * screenshotImage.naturalWidth;
        const y = (event.clientY - rect.top) / rect.height * screenshotImage.naturalHeight;

        corners.push({ x, y });
        drawCorners();

        if (corners.length === 4) {
            confirmAreaBtn.disabled = false;
        }

        // Aktualisiere die Meldung
        const messages = [
            "Bitte markieren Sie die erste Ecke (oben links).",
            "Bitte markieren Sie die zweite Ecke (oben rechts).",
            "Bitte markieren Sie die dritte Ecke (unten rechts).",
            "Bitte markieren Sie die vierte Ecke (unten links)."
        ];
        if (corners.length < 4) {
            cornerMessage.textContent = messages[corners.length];
        } else {
            cornerMessage.textContent = "";
        }
    });

    resetCornersBtn.addEventListener('click', () => {
        corners = [];
        drawCorners();
        confirmAreaBtn.disabled = true;
        cornerMessage.textContent = "Bitte markieren Sie die Ecken in der richtigen Reihenfolge.";
    });

    function drawCorners() {
        const context = screenshotCanvas.getContext('2d');
        const scaledWidth = screenshotCanvas.width;
        const scaledHeight = screenshotCanvas.height;

        // Lösche die alte Zeichnung
        context.clearRect(0, 0, scaledWidth, scaledHeight);

        // Zeichne die neuen Punkte
        corners.forEach((corner, index) => {
            const scaledX = corner.x / screenshotImage.naturalWidth * scaledWidth;
            const scaledY = corner.y / screenshotImage.naturalHeight * scaledHeight;

            context.beginPath();
            context.arc(scaledX, scaledY, 6, 0, 2 * Math.PI);
            context.fillStyle = '#ef4444';
            context.fill();
            context.lineWidth = 2;
            context.strokeStyle = 'white';
            context.stroke();

            context.font = '16px Inter';
            context.fillStyle = 'white';
            context.fillText(index + 1, scaledX + 10, scaledY - 10);
        });
    }

    confirmAreaBtn.addEventListener('click', () => {
        // Im realen Szenario: Senden der Ecken-Koordinaten an das Backend
        // fetch('/api/set_area', { method: 'POST', body: JSON.stringify({ corners }) });
        statusMessage.textContent = "Bereit zum Starten der Datenerfassung.";
        setView('processing');
    });

    startBtn.addEventListener('click', () => {
        // Im realen Szenario: Starten der Analyse über Websocket oder API-Aufruf
        // WebSocket-Verbindung herstellen und Start-Befehl senden
        // ws.send(JSON.stringify({ command: 'start' }));

        startBtn.disabled = true;
        stopBtn.disabled = false;
        statusMessage.textContent = "Datenerfassung läuft... Drücken Sie 'Stop', um die Heatmap zu generieren.";
    });

    stopBtn.addEventListener('click', () => {
        // Im realen Szenario: Senden des Stop-Befehls an das Backend
        // ws.send(JSON.stringify({ command: 'stop' }));

        startBtn.disabled = false;
        stopBtn.disabled = true;
        statusMessage.textContent = "Datenanalyse beendet. Erstelle Heatmap...";

        // Simuliere die Wartezeit und den Empfang der Heatmap-Daten
        setTimeout(() => {
            generateHeatmap();
            setView('heatmap');
        }, 2000);
    });

    restartBtn.addEventListener('click', () => {
        // App-Zustand zurücksetzen
        corners = [];
        confirmAreaBtn.disabled = true;
        cornerMessage.textContent = "Bitte markieren Sie die Ecken in der richtigen Reihenfolge.";
        rtmpUrlInput.value = 'rtmp://localhost/live/teststream';
        drawCorners(); // Canvas leeren
        setView('input');
    });

    modalCloseBtn.addEventListener('click', () => {
        modal.classList.add('hidden');
    });


    // Funktion zur Erstellung einer simulierten Heatmap
    function generateHeatmap() {
        const context = heatmapCanvas.getContext('2d');
        const width = heatmapCanvas.width;
        const height = heatmapCanvas.height;

        // Hintergrund
        context.fillStyle = 'black';
        context.fillRect(0, 0, width, height);

        // Simuliere Ballpositionen und erstelle die Heatmap-Daten
        const positions = [];
        for (let i = 0; i < 5000; i++) {
            positions.push({
                x: Math.random() * width,
                y: Math.random() * height,
            });
        }

        // Zeichne die Heatmap-Punkte
        positions.forEach(pos => {
            const gradient = context.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, 30);
            gradient.addColorStop(0, 'rgba(255, 0, 0, 0.4)');
            gradient.addColorStop(0.5, 'rgba(255, 255, 0, 0.2)');
            gradient.addColorStop(1, 'rgba(0, 0, 255, 0)');
            context.fillStyle = gradient;
            context.fillRect(pos.x - 30, pos.y - 30, 60, 60);
        });
    }

    // Stelle sicher, dass die Canvas-Größen korrekt sind
    window.addEventListener('resize', () => {
        screenshotCanvas.width = screenshotImage.width;
        screenshotCanvas.height = screenshotImage.height;
    });

    // Initialisiere die erste Ansicht
    setView('input');
</script>

</body>
</html>