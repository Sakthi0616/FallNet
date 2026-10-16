document.addEventListener("DOMContentLoaded", () => {
    const sampleSelect = document.getElementById("sample-select");
    const folderInput = document.getElementById("folder-input");
    const modelSelect = document.getElementById("model-select");
    const analyzeBtn = document.getElementById("analyze-btn");
    const btnText = analyzeBtn.querySelector(".btn-text");
    const spinner = analyzeBtn.querySelector(".spinner");

    const emptyState = document.getElementById("empty-state");
    const resultsContent = document.getElementById("results-content");
    const statusStep = document.getElementById("status-step");

    const alertBanner = document.getElementById("alert-banner");
    const alertIcon = document.getElementById("alert-icon");
    const alertTitle = document.getElementById("alert-title");
    const alertDesc = document.getElementById("alert-desc");

    const metricProb = document.getElementById("metric-prob");
    const metricThr = document.getElementById("metric-thr");
    const metricFrames = document.getElementById("metric-frames");
    const metricLatency = document.getElementById("metric-latency");

    const keyframeNum = document.getElementById("keyframe-num");
    const frameRgb = document.getElementById("frame-rgb");
    const frameMotion = document.getElementById("frame-motion");
    const timelineCanvas = document.getElementById("timeline-canvas");

    // Load available sample folders
    fetch("/api/samples")
        .then(res => res.json())
        .then(data => {
            if (data.samples) {
                data.samples.forEach(s => {
                    const opt = document.createElement("option");
                    opt.value = s.name;
                    opt.textContent = `${s.name} - ${s.label}`;
                    sampleSelect.appendChild(opt);
                });
            }
        })
        .catch(err => console.error("Failed to load samples:", err));

    // When a sample is selected, populate folder input
    sampleSelect.addEventListener("change", () => {
        if (sampleSelect.value) {
            folderInput.value = sampleSelect.value;
        }
    });

    // Run prediction
    analyzeBtn.addEventListener("click", () => {
        const folder = folderInput.value.trim();
        if (!folder) {
            alert("Please select a sample folder or type a folder name/path.");
            return;
        }

        const model = modelSelect.value;

        // UI Loading State
        analyzeBtn.disabled = true;
        btnText.textContent = "Analyzing Frames...";
        spinner.style.display = "inline-block";
        statusStep.textContent = "Processing...";

        fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ folder, model })
        })
        .then(res => {
            if (!res.ok) {
                return res.json().then(err => { throw new Error(err.error || "Prediction failed"); });
            }
            return res.json();
        })
        .then(data => {
            renderResults(data);
        })
        .catch(err => {
            alert("Error: " + err.message);
        })
        .finally(() => {
            analyzeBtn.disabled = false;
            btnText.textContent = "Analyze Thermal Video";
            spinner.style.display = "none";
        });
    });

    function renderResults(data) {
        emptyState.style.display = "none";
        resultsContent.style.display = "block";
        statusStep.textContent = "Analysis Complete";

        // 1. Alert Banner
        if (data.is_fall) {
            alertBanner.className = "alert-banner alert-danger";
            alertIcon.innerHTML = `
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
                    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                    <line x1="12" y1="9" x2="12" y2="13"/>
                    <line x1="12" y1="17" x2="12.01" y2="17"/>
                </svg>
            `;
            alertTitle.textContent = "CRITICAL: FALL DETECTED";
            alertDesc.textContent = `A severe sudden impact event was verified by the ${data.model_used}. Peak probability reached ${(data.peak_prob * 100).toFixed(1)}%.`;
        } else {
            alertBanner.className = "alert-banner alert-success";
            alertIcon.innerHTML = `
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                    <polyline points="9 12 11 14 15 10"/>
                </svg>
            `;
            alertTitle.textContent = "NORMAL ACTIVITY CONFIRMED";
            alertDesc.textContent = `Continuous routine motion detected. No fall patterns identified across ${data.total_windows} sliding windows.`;
        }

        // 2. Metrics
        metricProb.textContent = `${(data.peak_prob * 100).toFixed(1)}%`;
        metricThr.textContent = `${data.threshold.toFixed(4)}`;
        metricFrames.textContent = `${data.total_frames} / ${data.total_windows}`;
        metricLatency.textContent = `${data.latency_ms} ms`;

        // 3. Vision Inspector Keyframes
        keyframeNum.textContent = data.peak_frame;
        frameRgb.src = `data:image/jpeg;base64,${data.peak_rgb_b64}`;
        frameMotion.src = `data:image/jpeg;base64,${data.peak_motion_b64}`;

        // 4. Render Canvas Timeline
        drawTimeline(data.timeline, data.threshold);
    }

    function drawTimeline(timeline, threshold) {
        if (!timeline || timeline.length === 0) return;

        const ctx = timelineCanvas.getContext("2d");
        const w = timelineCanvas.width;
        const h = timelineCanvas.height;

        ctx.clearRect(0, 0, w, h);

        const padding = { top: 20, right: 20, bottom: 25, left: 40 };
        const chartW = w - padding.left - padding.right;
        const chartH = h - padding.top - padding.bottom;

        // Background grid lines
        ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
        ctx.lineWidth = 1;
        for (let yVal = 0; yVal <= 1.0; yVal += 0.25) {
            const y = padding.top + chartH * (1 - yVal);
            ctx.beginPath();
            ctx.moveTo(padding.left, y);
            ctx.lineTo(w - padding.right, y);
            ctx.stroke();

            // Y-axis label
            ctx.fillStyle = "#64748b";
            ctx.font = "10px Inter";
            ctx.textAlign = "right";
            ctx.fillText(`${(yVal * 100).toFixed(0)}%`, padding.left - 6, y + 3);
        }

        const n = timeline.length;
        const getX = i => padding.left + (i / Math.max(1, n - 1)) * chartW;
        const getY = p => padding.top + chartH * (1 - Math.min(1.0, Math.max(0.0, p)));

        // Draw gradient area under the curve
        const grad = ctx.createLinearGradient(0, padding.top, 0, h - padding.bottom);
        grad.addColorStop(0, "rgba(99, 102, 241, 0.45)");
        grad.addColorStop(1, "rgba(99, 102, 241, 0.0)");

        ctx.beginPath();
        ctx.moveTo(getX(0), h - padding.bottom);
        for (let i = 0; i < n; i++) {
            ctx.lineTo(getX(i), getY(timeline[i].prob));
        }
        ctx.lineTo(getX(n - 1), h - padding.bottom);
        ctx.closePath();
        ctx.fillStyle = grad;
        ctx.fill();

        // Highlight Fall Zones in Red
        for (let i = 0; i < n; i++) {
            if (timeline[i].prob >= threshold) {
                const x = getX(i);
                ctx.fillStyle = "rgba(239, 68, 68, 0.35)";
                ctx.fillRect(x - chartW / n / 2, padding.top, chartW / n, chartH);
            }
        }

        // Draw Line Curve
        ctx.beginPath();
        ctx.strokeStyle = "#818cf8";
        ctx.lineWidth = 2.5;
        for (let i = 0; i < n; i++) {
            const x = getX(i);
            const y = getY(timeline[i].prob);
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Draw Threshold Line
        const thrY = getY(threshold);
        ctx.strokeStyle = "#ef4444";
        ctx.lineWidth = 1.5;
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        ctx.moveTo(padding.left, thrY);
        ctx.lineTo(w - padding.right, thrY);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = "#ef4444";
        ctx.font = "bold 10px Inter";
        ctx.textAlign = "left";
        ctx.fillText(`Threshold: ${threshold.toFixed(3)}`, padding.left + 8, Math.max(padding.top + 12, thrY - 6));
    }
});
