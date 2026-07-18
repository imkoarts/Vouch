const path = require("node:path");
const fs = require("node:fs");
const { chromium } = require("playwright");

(async () => {
  const root = path.resolve(__dirname, "..");
  const output = path.resolve(process.argv[2] || path.join(root, "artifacts", "vouch-demo.webm"));
  const seconds = Number(process.argv[3] || 180);
  fs.mkdirSync(path.dirname(output), { recursive: true });
  const executablePath = process.env.PLAYWRIGHT_BROWSER_PATH;
  const browser = await chromium.launch({ headless: true, ...(executablePath ? { executablePath } : {}) });
  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 },
    recordVideo: { dir: path.dirname(output), size: { width: 1440, height: 900 } },
  });
  const page = await context.newPage();
  await page.goto(`file:///${path.join(root, "demo", "index.html").replaceAll("\\", "/")}`);
  await page.waitForTimeout(seconds * 1000);
  const video = page.video();
  await context.close();
  if (video) fs.renameSync(await video.path(), output);
  await browser.close();
  process.stdout.write(`${output}\n`);
})().catch((error) => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exitCode = 1;
});
