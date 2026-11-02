"use client";

import { useCallback, useEffect, useState } from "react";
import { getJson } from "./api";

type ApiState<T> = {
  path: string | null;
  retryVersion: number;
  data: T | null;
  error: string | null;
  loading: boolean;
};

export function useApi<T>(path: string | null) {
  const [retryVersion, setRetryVersion] = useState(0);
  const [state, setState] = useState<ApiState<T>>(() => ({
    path,
    retryVersion: 0,
    data: null,
    error: null,
    loading: Boolean(path),
  }));
  const retry = useCallback(() => setRetryVersion((version) => version + 1), []);

  useEffect(() => {
    if (!path) return;
    const controller = new AbortController();
    let active = true;
    getJson<T>(path, { signal: controller.signal })
      .then((value) => {
        if (active) setState({ path, retryVersion, data: value, error: null, loading: false });
      })
      .catch((reason: unknown) => {
        if (active && !controller.signal.aborted) {
          setState({
            path,
            retryVersion,
            data: null,
            error: reason instanceof Error ? reason.message : "Could not load this information.",
            loading: false,
          });
        }
      })
      .finally(() => { active = false; });
    return () => {
      active = false;
      controller.abort();
    };
  }, [path, retryVersion]);

  const matchesCurrentRequest = state.path === path && state.retryVersion === retryVersion;
  const visibleState = matchesCurrentRequest
    ? state
    : { data: null, error: null, loading: Boolean(path) };
  return { ...visibleState, retry };
}
