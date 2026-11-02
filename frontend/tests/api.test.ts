import { afterEach, describe, expect, it, vi } from "vitest";
import { getJson, labelName, postJson } from "../app/lib/api";

afterEach(() => vi.unstubAllGlobals());

describe("PaperLens API client", () => {
  it("uses the versioned API base and sends JSON for POST requests", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ label: 1 }) });
    vi.stubGlobal("fetch", fetchMock);

    await expect(postJson("/classifications", { title: "Title", abstract: "Abstract" })).resolves.toEqual({ label: 1 });
    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/api/v1/classifications",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ title: "Title", abstract: "Abstract" }),
        headers: { Accept: "application/json", "Content-Type": "application/json" },
        cache: "no-store",
      }),
    );
  });

  it("surfaces the API detail and retains a status message for non-JSON errors", async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({ ok: false, status: 503, json: async () => ({ detail: "Database unavailable" }) })
      .mockResolvedValueOnce({ ok: false, status: 404, json: async () => { throw new Error("not JSON"); } });
    vi.stubGlobal("fetch", fetchMock);

    await expect(getJson("/papers")).rejects.toThrow("Database unavailable");
    await expect(getJson("/papers/missing")).rejects.toThrow("Request failed (404)");
  });

  it("keeps all four original numeric label names", () => {
    expect([0, 1, 2, 3].map(labelName)).toEqual(["Algorithms", "NLP/AI", "Systems / Infra", "Vision"]);
    expect(labelName(9)).toBe("Unknown");
  });
});
