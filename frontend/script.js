document
  .getElementById("predict-form")
  .addEventListener("submit", async (e) => {
    e.preventDefault();

    const domainInput = document.getElementById("domain-input").value.trim();
    if (!domainInput) return;

    // UI Elements
    const btn = document.getElementById("analyze-btn");
    const loading = document.getElementById("loading");
    const results = document.getElementById("results");
    const errorCard = document.getElementById("error");
    const errorMsg = document.getElementById("error-msg");

    // Reset state
    btn.disabled = true;
    loading.classList.remove("hidden");
    results.classList.add("hidden");
    errorCard.classList.add("hidden");

    try {
      // Call FastAPI Backend
      const response = await fetch("http://localhost:8000/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ domain: domainInput }),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(
          errData.detail || "Failed to connect to the prediction API.",
        );
      }

      const data = await response.json();

      // 1. Render Label & Probability
      const labelBadge = document.getElementById("label-badge");
      labelBadge.textContent = data.label;
      labelBadge.className = `badge ${data.label.toLowerCase()}`;

      const probRaw = data.dga_probability;
      const probPercent = (probRaw * 100).toFixed(2);
      document.getElementById("prob-percentage").textContent =
        `${probPercent}%`;

      const probFill = document.getElementById("prob-fill");
      probFill.style.width = `${probPercent}%`;

      // Color the bar based on threshold
      if (probRaw >= 0.60) {
        probFill.style.backgroundColor = "var(--danger-color)";
      } else if (probRaw >= 0.40) {
        probFill.style.backgroundColor = "var(--warning-color)";
      } else {
        probFill.style.backgroundColor = "var(--success-color)";
      }

      // 1.5 Render Explanation Text Summary (V3 Upgrade)
      const explanationTextBox = document.getElementById("explanation-text");
      if (data.explanation_text) {
        explanationTextBox.textContent =
          "AI Reasoning: " + data.explanation_text;
        explanationTextBox.style.display = "block";
      } else {
        explanationTextBox.style.display = "none";
      }

      // 2. Render Explainability (Character Highlighting)
      const charBoxWrapper = document.getElementById("char-highlight-box");
      charBoxWrapper.innerHTML = "";

      // Find max absolute weight for normalization to control opacity safely
      let maxWeight = 0;
      data.explanation.forEach((item) => {
        if (Math.abs(item.weight) > maxWeight) {
          maxWeight = Math.abs(item.weight);
        }
      });

      // Prevent division by zero
      maxWeight = maxWeight === 0 ? 1 : maxWeight;

      data.explanation.forEach((item) => {
        const span = document.createElement("span");
        span.textContent = item.char;
        span.className = "char-box";

        // Normalize weight
        const rawWeight = item.weight;
        const normalized = Math.abs(rawWeight) / maxWeight;
        // Cap opacity to 0.9 for readability
        const alpha = Math.min(normalized, 0.9).toFixed(2);

        // Positive weight -> DGA (Red), Negative weight -> Legit (Green)
        if (rawWeight > 0) {
          span.style.backgroundColor = `rgba(248, 81, 73, ${alpha})`;
        } else if (rawWeight < 0) {
          span.style.backgroundColor = `rgba(63, 185, 80, ${alpha})`;
        }

        // Tooltip to show exact weight on hover
        span.title = `Char: '${item.char}'\nWeight: ${rawWeight.toFixed(4)}`;

        charBoxWrapper.appendChild(span);
      });

      results.classList.remove("hidden");
    } catch (err) {
      errorMsg.textContent = err.message;
      errorCard.classList.remove("hidden");
    } finally {
      btn.disabled = false;
      loading.classList.add("hidden");
    }
  });
