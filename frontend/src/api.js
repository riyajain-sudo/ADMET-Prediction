const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

// Throws an Error with a human-readable message on any non-2xx response,
// so components can just try/catch and show err.message directly.
export async function predict(smiles) {
  const response = await fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ smiles }),
  });

  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? `request failed (${response.status})`);
  }

  return response.json();
}
