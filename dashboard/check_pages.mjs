// Loads every page at desktop and phone width and reports console errors and sideways scroll.
import { existsSync, mkdirSync } from 'node:fs'
import puppeteer from 'puppeteer-core'

const chrome = ['C:/Program Files/Google/Chrome/Application/chrome.exe', 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe'].find(existsSync)
const base = process.argv[2] ?? 'http://localhost:4173'
const out = process.argv[3] ?? '../logs/shots'
mkdirSync(out, { recursive: true })
const pages = ['', 'cyclones', 'floods', 'earthquakes', 'events', 'models', 'live', 'event/assam-2026', 'event/yagi-2024', 'event/nagqu-2026']
const browser = await puppeteer.launch({ executablePath: chrome, headless: 'new', args: ['--use-gl=swiftshader', '--enable-webgl', '--ignore-gpu-blocklist'] })
let bad = 0
for (const width of [1440, 390]) {
  for (const path of pages) {
    const page = await browser.newPage()
    await page.setViewport({ width, height: 900 })
    const errors = []
    page.on('pageerror', (e) => errors.push(e.message))
    page.on('console', (m) => { if (m.type() === 'error' && !/Failed to load resource|ERR_|WebGL/.test(m.text())) errors.push(m.text()) })
    await page.goto(`${base}/#/${path}`, { waitUntil: 'domcontentloaded' })
    await new Promise((r) => setTimeout(r, 2500))
    const over = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)
    const blank = await page.evaluate(() => document.body.innerText.trim().length < 40)
    await page.screenshot({ path: `${out}/${width}-${path.replace('/', '_') || 'home'}.png`, fullPage: path === 'models' || path === 'cyclones' })
    const ok = !errors.length && over <= 0 && !blank
    if (!ok) bad++
    console.log(ok ? 'ok ' : 'BAD', width, path || 'home', over > 0 ? `sideways ${over}px` : '', blank ? 'blank' : '', errors.slice(0, 2).join(' | ').slice(0, 200))
    await page.close()
  }
}
await browser.close()
console.log(bad ? `${bad} pages failed` : 'all pages passed')
process.exit(bad ? 1 : 0)
