import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { getJson } from "../app/lib/api";
import { useApi } from "../app/lib/use-api";

vi.mock("../app/lib/api", () => ({ getJson: vi.fn() }));

const getJsonMock = vi.mocked(getJson);

beforeEach(() => getJsonMock.mockReset());

describe("useApi", () => {
  it("loads typed data and provides an explicit retry action", async () => {
    getJsonMock.mockResolvedValueOnce({ paper_count: 3 } as never);
    const { result } = renderHook(() => useApi<{ paper_count: number }>("/overview"));

    expect(result.current.loading).toBe(true);
    await waitFor(() => expect(result.current.data).toEqual({ paper_count: 3 }));
    expect(result.current.error).toBeNull();
    expect(getJsonMock).toHaveBeenCalledWith("/overview", { signal: expect.any(AbortSignal) });

    getJsonMock.mockResolvedValueOnce({ paper_count: 4 } as never);
    act(() => result.current.retry());
    await waitFor(() => expect(result.current.data).toEqual({ paper_count: 4 }));
    expect(getJsonMock).toHaveBeenCalledTimes(2);
  });

  it("clears stale data when a request path changes and ignores an older response", async () => {
    const signals = new Map<string, AbortSignal>();
    getJsonMock.mockImplementation((path, init) => {
      if (init?.signal) signals.set(path, init.signal as AbortSignal);
      if (path === "/papers/old") return new Promise(() => {}) as never;
      return Promise.resolve({ id: "new" }) as never;
    });
    const { result, rerender } = renderHook(({ path }: { path: string }) => useApi<{ id: string }>(path), {
      initialProps: { path: "/papers/old" },
    });

    await waitFor(() => expect(signals.has("/papers/old")).toBe(true));
    rerender({ path: "/papers/new" });
    await waitFor(() => expect(result.current.data).toEqual({ id: "new" }));
    expect(signals.get("/papers/old")?.aborted).toBe(true);
  });
});
