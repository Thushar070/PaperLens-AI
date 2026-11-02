import { access, cp, mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

async function copyIfPresent(relativeSource, relativeDestination) {
  const source = resolve(projectRoot, relativeSource);
  const destination = resolve(projectRoot, relativeDestination);
  try {
    await access(source);
  } catch (error) {
    if (error?.code === "ENOENT") return;
    throw error;
  }

  await mkdir(dirname(destination), { recursive: true });
  await cp(source, destination, { recursive: true, force: true });
}

await copyIfPresent("public", ".next/standalone/public");
await copyIfPresent(".next/static", ".next/standalone/.next/static");
