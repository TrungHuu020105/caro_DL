import type { Session } from "./types";

export const API_BASE = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");

export class APIError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message);
    this.name = "APIError";
  }
}

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
      throw new APIError(message, response.status);
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

export function socketUrl(id: string) {
  const base = API_BASE ? new URL(API_BASE) : new URL(window.location.origin);
  base.protocol = base.protocol === "https:" ? "wss:" : "ws:";
  base.pathname = `${base.pathname.replace(/\/$/, "")}/ws/games/${id}`;
  return base.toString();
}

export function readRoomSession(): Session | null {
  try {
    const value = JSON.parse(
      sessionStorage.getItem("caro.active-session") ||
        sessionStorage.getItem("caro.room-session") ||
        "null",
    );
    return value &&
      typeof value.id === "string" &&
      typeof value.token === "string"
      ? value
      : null;
  } catch {
    return null;
  }
}

export function saveRoomSession(value: Session | null, online = true) {
  try {
    if (value)
      sessionStorage.setItem("caro.active-session", JSON.stringify(value));
    else sessionStorage.removeItem("caro.active-session");
    if (value && online)
      sessionStorage.setItem("caro.room-session", JSON.stringify(value));
    else sessionStorage.removeItem("caro.room-session");
  } catch {
    /* Room remains playable when browser storage is unavailable. */
  }
}
export function save(key: string, value: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* Private mode/storage full: game remains usable. */
  }
}
