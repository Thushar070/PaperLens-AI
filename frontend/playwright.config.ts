import { defineConfig, devices } from "@playwright/test";

const baseURL = "http://127.0.0.1:3109";

export default defineConfig({
  testDir: "./tests/e2e",
  fullyParallel: true,
  workers: 2,
  timeout: 60_000,
  reporter: "list",
  outputDir: "../artifacts/phase9a/playwright-results",
  use: {
    ...devices["Desktop Chrome"],
    channel: "msedge",
    baseURL,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  webServer: {
    command: "node .next/standalone/server.js",
    url: baseURL,
    reuseExistingServer: false,
    timeout: 120_000,
    env: {
      NEXT_TELEMETRY_DISABLED: "1",
      HOSTNAME: "127.0.0.1",
      PORT: "3109",
      NEXT_PUBLIC_API_BASE_URL: "http://127.0.0.1:8001/api/v1",
    },
  },
});
