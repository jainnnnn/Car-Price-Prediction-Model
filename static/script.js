(function () {
  const form = document.getElementById("car-form");
  const submitBtn = document.getElementById("submit-btn");
  const resultPanel = document.getElementById("result");

  const states = {
    empty: document.getElementById("state-empty"),
    error: document.getElementById("state-error"),
    result: document.getElementById("state-result"),
  };

  const ruler = document.querySelector(".ruler");
  const rulerMin = parseFloat(ruler.dataset.min);
  const rulerMax = parseFloat(ruler.dataset.max);

  const usd = new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  });

  function show(name) {
    Object.entries(states).forEach(([key, el]) => { el.hidden = key !== name; });
  }

  function pct(value) {
    const p = ((value - rulerMin) / (rulerMax - rulerMin)) * 100;
    return Math.min(100, Math.max(0, p));
  }

  function showError(messages) {
    const list = document.getElementById("error-list");
    list.innerHTML = "";
    messages.forEach((m) => {
      const li = document.createElement("li");
      li.textContent = m;
      list.appendChild(li);
    });
    show("error");
  }

  function showResult(data) {
    document.getElementById("price-main").textContent = usd.format(data.price);
    document.getElementById("price-low").textContent = usd.format(data.low);
    document.getElementById("price-high").textContent = usd.format(data.high);

    const low = pct(data.low);
    const high = pct(data.high);
    const band = document.getElementById("ruler-band");
    band.style.left = low + "%";
    band.style.width = Math.max(high - low, 1) + "%";
    document.getElementById("ruler-marker").style.left = pct(data.price) + "%";

    const notes = document.getElementById("notes");
    notes.innerHTML = "";
    (data.notes || []).forEach((n) => {
      const li = document.createElement("li");
      li.textContent = n;
      notes.appendChild(li);
    });
    show("result");
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    form.querySelectorAll("[aria-invalid]").forEach((el) => el.removeAttribute("aria-invalid"));

    const payload = Object.fromEntries(new FormData(form).entries());

    submitBtn.disabled = true;
    submitBtn.textContent = "Estimating…";

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();

      if (!response.ok || !data.ok) {
        showError(data.errors || ["Something went wrong. Try again."]);
      } else {
        showResult(data);
      }
    } catch (err) {
      showError(["Could not reach the server. Check that the Flask app is running."]);
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = "Estimate price";
    }

    // On narrow screens the result sits below the form, so bring it into view
    if (window.matchMedia("(max-width: 860px)").matches) {
      const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      resultPanel.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "start" });
    }
  });

  form.addEventListener("reset", () => show("empty"));
})();
