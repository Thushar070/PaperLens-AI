import AxeBuilder from "@axe-core/playwright";
import { expect, test } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { mockDatabaseApi, samplePaper, savedSearchQuery } from "./phase9-fixtures";

const screenshotDirectory = resolve(process.cwd(), "../artifacts/phase9a/screenshots");
mkdirSync(screenshotDirectory, { recursive: true });

const routes = [
  { path: "/", heading: /Discover, classify and explore research papers/i, active: "Home" },
  { path: "/papers", heading: "Browse papers", active: "Browse Papers" },
  { path: `/papers/${encodeURIComponent(samplePaper.arxiv_id)}`, heading: samplePaper.title, active: "Browse Papers" },
  { path: "/classify", heading: "Classify a paper", active: "Classification" },
  { path: "/search", heading: "Search research papers", active: "Semantic Search" },
  { path: "/experiments", heading: "Experiment comparison", active: "Experiments" },
];

test.beforeEach(async ({ page }) => {
  await mockDatabaseApi(page);
});

test("all six routes render directly and expose the correct active navigation", async ({ page }) => {
  const consoleErrors: string[] = [];
  const pageErrors: string[] = [];
  const failedRequests: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") consoleErrors.push(message.text());
  });
  page.on("pageerror", (error) => pageErrors.push(error.message));
  page.on("requestfailed", (request) => {
    const errorText = request.failure()?.errorText ?? "unknown request failure";
    // Next cancels speculative RSC prefetches when the smoke test navigates away.
    if (!errorText.includes("ERR_ABORTED")) failedRequests.push(`${request.method()} ${request.url()}: ${errorText}`);
  });

  for (const route of routes) {
    await page.goto(route.path);
    await expect(page.getByRole("heading", { level: 1, name: route.heading })).toBeVisible();
    await expect(page.getByRole("link", { name: route.active, exact: true })).toHaveAttribute("aria-current", "page");
    const screenshotName = route.path === "/" ? "home" : route.path.startsWith("/papers/") ? "paper-details" : route.active.toLowerCase().replaceAll(" ", "-");
    await page.screenshot({ path: resolve(screenshotDirectory, `${screenshotName}.png`), fullPage: true });
  }

  expect(pageErrors).toEqual([]);
  expect(consoleErrors).toEqual([]);
  expect(failedRequests).toEqual([]);
});

test("sidebar navigation and paper detail links work", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("link", { name: "Browse Papers", exact: true }).click();
  await expect(page).toHaveURL(/\/papers$/);
  await expect(page.getByRole("heading", { name: "Browse papers" })).toBeVisible();

  await page.getByRole("link", { name: `View ${samplePaper.title}`, exact: true }).click();
  await expect(page).toHaveURL(new RegExp(`/papers/${encodeURIComponent(samplePaper.arxiv_id)}$`));
  await expect(page.getByRole("heading", { level: 1, name: samplePaper.title })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Abstract" })).toBeVisible();
});

test("paper browsing filters by text and class and paginates real Phase 1 rows", async ({ page }) => {
  await page.goto("/papers");
  await expect(page.getByText("19,999 papers found")).toBeVisible();
  await expect(page.getByRole("link", { name: samplePaper.title, exact: true })).toBeVisible();

  await page.getByRole("button", { name: "Next" }).click();
  await expect(page.getByText("Showing 21–40 of 19,999")).toBeVisible();
  await page.getByRole("button", { name: "Previous" }).click();
  await expect(page.getByText("Showing 1–20 of 19,999")).toBeVisible();

  const firstTerm = samplePaper.title.split(/\s+/)[0];
  await page.getByPlaceholder("Search titles and abstracts…").fill(firstTerm);
  await page.getByRole("button", { name: "Search" }).click();
  await expect(page.getByRole("link", { name: samplePaper.title, exact: true })).toBeVisible();

  await page.getByLabel("Filter by research category").selectOption(String(samplePaper.label));
  await page.getByRole("button", { name: "Search" }).click();
  await expect(page.getByRole("link", { name: samplePaper.title, exact: true })).toBeVisible();
  const displayedCategories = await page.locator("tbody .category-badge").allTextContents();
  expect(displayedCategories.length).toBeGreaterThan(0);
  expect(new Set(displayedCategories).size).toBe(1);
});

test("classification validates both fields and displays the saved-model response", async ({ page }) => {
  await page.goto("/classify");
  const submit = page.getByRole("button", { name: "Classify paper" });
  await expect(submit).toBeDisabled();
  await page.getByLabel("Title").fill(samplePaper.title);
  await expect(submit).toBeDisabled();
  await page.getByLabel("Abstract").fill(samplePaper.abstract);
  await expect(submit).toBeEnabled();
  await submit.click();
  await expect(page.getByText("P3/tfidf_svm")).toBeVisible({ timeout: 30_000 });
  await expect(page.getByText(/Decision scores/)).toBeVisible();
  await expect(page.getByText(/not calibrated probabilities/i)).toBeVisible();
  await expect(page.locator(".prediction-category .category-badge")).toContainText(/Algorithms|NLP\/AI|Systems \/ Infra|Vision/);
});

test("search submits the selected method, shows saved ranked records, and handles empty results", async ({ page }) => {
  await page.goto("/search");
  const submit = page.getByRole("button", { name: "Search" });
  await expect(submit).toBeDisabled();
  await page.getByLabel("What are you researching?").fill(savedSearchQuery);
  await expect(submit).toBeEnabled();
  await submit.click();
  await expect(page.getByRole("heading", { name: "5 matching papers" })).toBeVisible();
  await expect(page.getByText("Scores are ranking signals, not probabilities")).toBeVisible();
  await expect(page.getByText(/assistant-assessed document judgments; those sparse judgments were not human-reviewed/i)).toBeVisible();

  await page.getByRole("radio", { name: /MiniLM Phase 7 encoder/ }).check();
  await submit.click();
  await expect(page.getByText("PHASE 7 MINILM", { exact: true })).toBeVisible();

  const emptyPage = await page.context().newPage();
  await mockDatabaseApi(emptyPage, { emptySearch: true });
  await emptyPage.goto("/search");
  await emptyPage.getByLabel("What are you researching?").fill(savedSearchQuery);
  await emptyPage.getByRole("button", { name: "Search" }).click();
  await expect(emptyPage.getByRole("heading", { name: "No matching papers found" })).toBeVisible();
  await emptyPage.close();
});

test("API failures are shown and retry actions issue a fresh request", async ({ page }) => {
  await page.unroute("**/api/v1/**");
  await mockDatabaseApi(page, { paperListFailures: 1 });
  await page.goto("/papers");
  await expect(page.locator(".notice-error[role=alert]")).toContainText("Paper database is unavailable");
  await page.getByRole("button", { name: "Try again" }).click();
  await expect(page.getByRole("link", { name: samplePaper.title, exact: true })).toBeVisible();

  const searchPage = await page.context().newPage();
  await mockDatabaseApi(searchPage, { searchFailures: 1 });
  await searchPage.goto("/search");
  await searchPage.getByLabel("What are you researching?").fill(savedSearchQuery);
  await searchPage.getByRole("button", { name: "Search" }).click();
  await expect(searchPage.locator(".notice-error[role=alert]")).toContainText("Search could not complete");
  await searchPage.getByRole("button", { name: "Search" }).click();
  await expect(searchPage.getByRole("heading", { name: "5 matching papers" })).toBeVisible();
  await searchPage.close();
});

test("desktop, tablet, and mobile layouts do not overflow the viewport", async ({ page }) => {
  for (const viewport of [{ width: 1440, height: 1000 }, { width: 768, height: 1024 }, { width: 390, height: 844 }]) {
    await page.setViewportSize(viewport);
    for (const route of routes) {
      await page.goto(route.path);
      await expect(page.getByRole("heading", { level: 1, name: route.heading })).toBeVisible();
      const dimensions = await page.evaluate(() => ({
        viewportWidth: document.documentElement.clientWidth,
        documentWidth: document.documentElement.scrollWidth,
        overflowingElements: Array.from(document.querySelectorAll("body *"))
          .filter((element) => {
            const bounds = element.getBoundingClientRect();
            return bounds.width > 0 && (bounds.left < -1 || bounds.right > window.innerWidth + 1);
          })
          .slice(0, 12)
          .map((element) => {
            const bounds = element.getBoundingClientRect();
            return {
              tag: element.tagName,
              className: typeof element.className === "string" ? element.className : "",
              left: Math.round(bounds.left),
              right: Math.round(bounds.right),
              width: Math.round(bounds.width),
            };
          }),
        layout: [".content-column", "#main-content", ".browse-card", ".table-scroll", ".paper-table"].map((selector) => {
          const element = document.querySelector(selector);
          if (!element) return { selector, found: false };
          const bounds = element.getBoundingClientRect();
          const style = getComputedStyle(element);
          return {
            selector,
            found: true,
            left: Math.round(bounds.left),
            right: Math.round(bounds.right),
            width: Math.round(bounds.width),
            clientWidth: element.clientWidth,
            scrollWidth: element.scrollWidth,
            minWidth: style.minWidth,
            overflowX: style.overflowX,
          };
        }),
      }));
      expect(dimensions.documentWidth, `${route.path} overflows at ${viewport.width}px: ${JSON.stringify({ elements: dimensions.overflowingElements, layout: dimensions.layout })}`).toBeLessThanOrEqual(dimensions.viewportWidth + 1);
      if (viewport.width === 390) {
        const screenshotName = route.path === "/" ? "home" : route.path.startsWith("/papers/") ? "paper-details" : route.active.toLowerCase().replaceAll(" ", "-");
        await page.screenshot({ path: resolve(screenshotDirectory, `${screenshotName}-mobile.png`), fullPage: true });
      }
    }
  }
});

test("all six routes have no WCAG 2.1 A/AA axe violations", async ({ page }) => {
  const violations = new Map<string, {
    routes: string[];
    id: string;
    impact: string | null | undefined;
    description: string;
    target: unknown;
    summary?: string;
  }>();
  for (const route of routes) {
    await page.goto(route.path);
    await expect(page.getByRole("heading", { level: 1, name: route.heading })).toBeVisible();
    const result = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]).analyze();
    for (const violation of result.violations) {
      for (const node of violation.nodes) {
        const target = JSON.stringify(node.target);
        const key = `${violation.id}:${target}`;
        const existing = violations.get(key);
        if (existing) {
          if (!existing.routes.includes(route.path)) existing.routes.push(route.path);
        } else {
          violations.set(key, {
            routes: [route.path],
            id: violation.id,
            impact: violation.impact,
            description: violation.help,
            target: node.target,
            summary: node.failureSummary,
          });
        }
      }
    }
  }
  const report = Array.from(violations.values());
  writeFileSync(resolve(screenshotDirectory, "..", "axe-report.json"), JSON.stringify({
    standard: "WCAG 2.1 A/AA",
    routes: routes.map(({ path }) => path),
    uniqueViolationCount: report.length,
    violations: report,
  }, null, 2));
  expect(report).toEqual([]);
});
