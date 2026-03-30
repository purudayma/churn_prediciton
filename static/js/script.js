/* ═══════════════════════════════════════════
   ChurnIQ — script.js
═══════════════════════════════════════════ */

document.addEventListener("DOMContentLoaded", () => {

  // ── 1. Scroll reveal ──────────────────────────────────────────
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add("visible");
        observer.unobserve(e.target);
      }
    });
  }, { threshold: 0.12 });

  document.querySelectorAll(
    ".feat-card, .step, .kpi-card, .chart-card, .form-card, .result-card, .reco-card"
  ).forEach((el, i) => {
    el.classList.add("reveal");
    el.style.transitionDelay = `${i * 0.06}s`;
    observer.observe(el);
  });

  // ── 2. Form submission loading state ──────────────────────────
  const form = document.getElementById("predictForm");
  if (form) {
    form.addEventListener("submit", (e) => {
      if (!validateForm(e)) return;
      const btnText   = document.getElementById("btnText");
      const btnLoader = document.getElementById("btnLoader");
      const submitBtn = document.getElementById("submitBtn");
      if (btnText && btnLoader) {
        btnText.classList.add("hidden");
        btnLoader.classList.remove("hidden");
        submitBtn.disabled = true;
      }
    });
  }

  // ── 3. Form validation ─────────────────────────────────────────
  function validateForm(e) {
    let valid = true;
    const form = document.getElementById("predictForm");
    form.querySelectorAll("input[required], select[required]").forEach(field => {
      field.classList.remove("error");
      if (!field.value.trim()) {
        field.classList.add("error");
        valid = false;
      }
      // Range checks
      const min = parseFloat(field.min);
      const max = parseFloat(field.max);
      const val = parseFloat(field.value);
      if (!isNaN(min) && !isNaN(max) && (val < min || val > max)) {
        field.classList.add("error");
        valid = false;
      }
    });
    if (!valid) {
      e.preventDefault();
      const first = form.querySelector(".error");
      if (first) first.scrollIntoView({ behavior: "smooth", block: "center" });
      showToast("⚠️ Please fix the highlighted fields.", "warning");
    }
    return valid;
  }

  // ── 4. Auto-calculate total spending ──────────────────────────
  const ordersInput = document.getElementById("number_of_orders");
  const aovInput    = document.getElementById("avg_order_value");
  const totalInput  = document.getElementById("total_spending");

  function calcTotal() {
    if (!ordersInput || !aovInput || !totalInput) return;
    const orders = parseFloat(ordersInput.value) || 0;
    const aov    = parseFloat(aovInput.value) || 0;
    if (orders > 0 && aov > 0 && !totalInput.dataset.userEdited) {
      totalInput.value = (orders * aov).toFixed(2);
    }
  }
  if (ordersInput) ordersInput.addEventListener("input", calcTotal);
  if (aovInput)    aovInput.addEventListener("input", calcTotal);
  if (totalInput) {
    totalInput.addEventListener("input", () => {
      totalInput.dataset.userEdited = "1";
    });
  }

  // ── 5. Result page: animate probability bars ───────────────────
  document.querySelectorAll(".prob-bar[data-width]").forEach(bar => {
    setTimeout(() => {
      bar.style.width = bar.dataset.width + "%";
    }, 400);
  });

  // Count-up for probability numbers
  document.querySelectorAll(".prob-num[data-target]").forEach(el => {
    const target = parseFloat(el.dataset.target);
    let current = 0;
    const step  = target / 60;
    const timer = setInterval(() => {
      current += step;
      if (current >= target) { current = target; clearInterval(timer); }
      el.textContent = current.toFixed(1) + "%";
    }, 25);
  });

  // ── 6. Toast notifications ─────────────────────────────────────
  function showToast(msg, type = "info") {
    const existing = document.querySelector(".toast");
    if (existing) existing.remove();

    const toast = document.createElement("div");
    toast.className = "toast";
    toast.textContent = msg;
    Object.assign(toast.style, {
      position: "fixed", bottom: "24px", left: "50%",
      transform: "translateX(-50%) translateY(20px)",
      background: type === "warning" ? "rgba(245,158,11,.15)" : "rgba(124,58,237,.2)",
      border: `1px solid ${type === "warning" ? "rgba(245,158,11,.4)" : "rgba(124,58,237,.4)"}`,
      color: "#e2e8f0",
      padding: "12px 24px", borderRadius: "10px",
      fontSize: ".9rem", fontFamily: "'DM Sans', sans-serif",
      backdropFilter: "blur(8px)",
      zIndex: "9999",
      transition: "all .3s ease",
      opacity: "0",
    });
    document.body.appendChild(toast);
    requestAnimationFrame(() => {
      toast.style.opacity = "1";
      toast.style.transform = "translateX(-50%) translateY(0)";
    });
    setTimeout(() => {
      toast.style.opacity = "0";
      setTimeout(() => toast.remove(), 300);
    }, 3500);
  }

  // ── 7. Nav active highlight on scroll ─────────────────────────
  const sections = document.querySelectorAll("section[id]");
  if (sections.length) {
    const navObserver = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          document.querySelectorAll(".nav-links a").forEach(a => {
            a.classList.toggle("active", a.getAttribute("href") === "#" + e.target.id);
          });
        }
      });
    }, { threshold: 0.4 });
    sections.forEach(s => navObserver.observe(s));
  }

  // ── 8. Input: remove error on change ──────────────────────────
  document.querySelectorAll("input, select").forEach(el => {
    el.addEventListener("input", () => el.classList.remove("error"));
  });

});
