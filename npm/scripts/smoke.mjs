#!/usr/bin/env node
// End-to-end check of the npm packaging on the host platform: build the
// packages from target/release/<NAME>, `npm pack` them, install the tarballs
// into a clean project (no registry), run the launcher, and speak MCP to it.
import { execFileSync, spawn } from "node:child_process";
import { copyFileSync, mkdirSync, mkdtempSync, readdirSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { buildPackages, NAME, PLATFORMS, workspaceVersion } from "./build-packages.mjs";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const win = process.platform === "win32";
const npm = win ? "npm.cmd" : "npm";

function hostPlatform() {
  const cpu = { arm64: "arm64", x64: "x64" }[process.arch];
  const header = process.report?.getReport().header;
  const libc = header?.glibcVersionRuntime ? "gnu" : "musl";
  const suffix =
    process.platform === "darwin" ? `darwin-${cpu}`
    : process.platform === "win32" ? `win32-${cpu}-msvc`
    : `linux-${cpu}-${libc}`;
  const p = PLATFORMS.find((x) => x.suffix === suffix);
  if (!p) throw new Error(`no packaging target for ${suffix}`);
  return p;
}

const tmp = mkdtempSync(path.join(os.tmpdir(), `${NAME}-npm-smoke-`));
const version = workspaceVersion();
const host = hostPlatform();
const exe = win ? `${NAME}.exe` : NAME;

const binaries = path.join(tmp, "binaries", host.target);
mkdirSync(binaries, { recursive: true });
copyFileSync(path.join(ROOT, "target/release", exe), path.join(binaries, exe));

const out = path.join(tmp, "packages");
const built = buildPackages({ binaries: path.join(tmp, "binaries"), out, version, targets: [host.suffix] });

const tarballs = path.join(tmp, "tarballs");
mkdirSync(tarballs);
const pack = (dir) =>
  execFileSync(npm, ["pack", dir, "--pack-destination", tarballs, "--silent"], { encoding: "utf8" }).trim().split("\n").pop();
const platformTar = path.join(tarballs, pack(built.platforms[0]));
const rootTar = path.join(tarballs, pack(built.root));

const project = path.join(tmp, "project");
mkdirSync(project);
writeFileSync(
  path.join(project, "package.json"),
  JSON.stringify({
    name: "smoke",
    private: true,
    dependencies: { [`@krv-labs/${NAME}`]: `file:${rootTar}` },
    overrides: { [`@krv-labs/${NAME}-${host.suffix}`]: `file:${platformTar}` },
  }),
);
execFileSync(npm, ["install", "--no-audit", "--no-fund", "--loglevel=error"], { cwd: project, stdio: "inherit" });

const launcher = path.join(project, `node_modules/@krv-labs/${NAME}/bin/${NAME}.js`);
const versionOut = execFileSync(process.execPath, [launcher, "--version"], { encoding: "utf8" });
if (!versionOut.includes(version)) throw new Error(`--version printed ${JSON.stringify(versionOut)}, expected ${version}`);
console.log(`launcher ok: ${versionOut.trim()}`);

// MCP handshake over stdio through the launcher.
const server = spawn(process.execPath, [launcher, "mcp"], {
  stdio: ["pipe", "pipe", "inherit"],
  env: { ...process.env, PULSAR_STORE_DIR: path.join(tmp, "store") },
});
const pending = new Map();
let buffer = "";
server.stdout.on("data", (chunk) => {
  buffer += chunk;
  for (let nl; (nl = buffer.indexOf("\n")) >= 0; ) {
    const line = buffer.slice(0, nl).trim();
    buffer = buffer.slice(nl + 1);
    if (!line) continue;
    const msg = JSON.parse(line);
    pending.get(msg.id)?.(msg);
  }
});
const send = (msg) => server.stdin.write(JSON.stringify(msg) + "\n");
const request = (id, method, params) =>
  new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`timeout waiting for ${method}`)), 20_000);
    pending.set(id, (m) => { clearTimeout(timer); resolve(m); });
    send({ jsonrpc: "2.0", id, method, params });
  });

try {
  const init = await request(1, "initialize", {
    protocolVersion: "2025-06-18",
    capabilities: {},
    clientInfo: { name: "npm-smoke", version: "0" },
  });
  if (!init.result?.serverInfo) throw new Error(`bad initialize response: ${JSON.stringify(init)}`);
  send({ jsonrpc: "2.0", method: "notifications/initialized" });
  const tools = await request(2, "tools/list", {});
  const names = tools.result.tools.map((t) => t.name);
  for (const expected of ["ingest_dataset", "run_topological_sweep", "get_workflow_guide"]) {
    if (!names.includes(expected)) throw new Error(`tools/list is missing ${expected}`);
  }
  console.log(`mcp ok: ${init.result.serverInfo.name} ${init.result.serverInfo.version}, ${names.length} tools`);
} finally {
  server.kill();
}
console.log(`packages: ${readdirSync(tarballs).join(", ")}`);
