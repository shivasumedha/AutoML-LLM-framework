// Single place for all backend calls.
// In production, VITE_API_URL points at the Render FastAPI service.
// For local Vite development, point directly at the local API by default.
export const API_URL = import.meta.env.VITE_API_URL || (import.meta.env.PROD ? "/api" : "http://127.0.0.1:8000");

function formatDetail(detail) {
  if (!detail) return "";
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
  }
  return JSON.stringify(detail);
}

/**
 * api("/dataset")
 * api("/train", { method: "POST", body: { target_column: "species" }, timeout: 120000 })
 * api("/upload-dataset", { method: "POST", form: formData })
 *
 * Always settles: it resolves with JSON or throws an Error with a readable
 * message. A timeout guarantees the UI can never hang on "Uploading...".
 */
export async function api(path, { method = "GET", body, form, timeout = 30000 } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);

  try {
    const options = { method, signal: controller.signal };
    const sessionId = localStorage.getItem("automl-chat-session");
    if (sessionId) options.headers = { "X-Chat-Session": sessionId };

    if (form) {
      // Do NOT set Content-Type here: the browser adds the multipart boundary.
      options.body = form;
    } else if (body !== undefined) {
      options.headers = { ...options.headers, "Content-Type": "application/json" };
      options.body = JSON.stringify(body);
    }

    const res = await fetch(`${API_URL}${path}`, options);

    let data = null;
    try {
      data = await res.json();
    } catch {
      /* non-JSON response */
    }

    if (!res.ok) {
      throw new Error(formatDetail(data?.detail) || `Request failed (${res.status})`);
    }
    return data;
  } catch (err) {
    if (err.name === "AbortError") {
      throw new Error(
        `Request timed out after ${Math.round(timeout / 1000)}s. Check that the backend is running at ${API_URL}.`
      );
    }
    if (err instanceof TypeError) {
      throw new Error(
        `Cannot reach the backend at ${API_URL}. Start it with: uvicorn backend.main:app --reload --port 8000`
      );
    }
    throw err;
  } finally {
    clearTimeout(timer);
  }
}
