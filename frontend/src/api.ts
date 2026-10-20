import type { Session } from "./types";

export const API_BASE = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");

export async function request<T>(
  path: string,
  options: RequestInit = {},
  token?: string,
): Promise<T> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 30000);
  try {
    const response = await fetch(`${API_BASE}/api${path}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...(token ? { "X-Player-Token": token } : {}),
        ...options.headers,
      },
      signal: controller.signal,
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      const message =
        typeof body.detail === "string"
          ? body.detail
          : "Dữ liệu chưa hợp lệ. Hãy kiểm tra thiết lập.";
      throw new Error(message);
    }
    return (await response.json()) as T;
  } catch (error) {
    if (
      error instanceof TypeError ||
      (error instanceof DOMException && error.name === "AbortError")
    ) {
      throw new Error(
        "Không kết nối được máy chủ. Hãy kiểm tra backend ở cổng 8000 rồi thử lại.",
      );
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}

export function gameRequest<T>(
  session: Session,
  action = "",
  body?: unknown,
): Promise<T> {
  return request<T>(
    `/games/${session.id}${action ? `/${action}` : ""}`,
    {
      method: action ? "POST" : "GET",
      ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
    },
    session.token,
  );
}

export function readSaved<T>(key: string, fallback: T): T {
  try {
    return JSON.parse(localStorage.getItem(key) || "null") ?? fallback;
  } catch {
    return fallback;
  }
}
export function save(key: string, value: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* Private mode/storage full: game remains usable. */
  }
}
