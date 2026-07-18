const path = require("node:path");
const fs = require("node:fs");
const { chromium } = require("playwright");

(async () => {
  const root = path.resolve(__dirname, "..");
  const output = path.resolve(process.argv[2] || path.join(root, "artifacts", "demo-frames"));
  fs.mkdirSync(output, { recursive: true });
  const executablePath = process.env.PLAYWRIGHT_BROWSER_PATH;
  const browser = await chromium.launch({
    headless: true,
    ...(executablePath ? { executablePath } : {}),
  });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  await page.goto(`file:///${path.join(root, "demo", "index.html").replaceAll("\\", "/")}`);
  for (const seconds of [5, 25, 55, 90, 115, 140, 165]) {
    await page.evaluate((milliseconds) => {
      for (const animation of document.getAnimations()) {
        animation.pause();
        animation.currentTime = milliseconds;
      }
    }, seconds * 1000);
    await page.screenshot({ path: path.join(output, `${seconds}.png`), scale: "css" });
  }
  const fit = await page.evaluate(() => ({
    width: innerWidth,
    height: innerHeight,
    scrollWidth: document.documentElement.scrollWidth,
    scrollHeight: document.documentElement.scrollHeight,
  }));
  await browser.close();
  process.stdout.write(`${JSON.stringify(fit)}\n`);
})().catch((error) => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exitCode = 1;
});
