#!/usr/bin/env node
// Launcher for the Pulsar binary. The binary itself ships in a per-platform
// package (@krv-labs/<NAME>-<platform>) that the package manager installed as
// an optional dependency; nothing is downloaded at install or run time.
"use strict";

const { spawn } = require("node:child_process");
const path = require("node:path");

const SCOPE = "@krv-labs";
// Stamped by npm/scripts/build-packages.mjs from its NAME.
const NAME = "pulsar";

function libc() {
  if (process.platform !== "linux") return null;
  const header = process.report && process.report.getReport().header;
  return header && header.glibcVersionRuntime ? "gnu" : "musl";
}

function platformSuffix() {
  const { platform, arch } = process;
  const cpu = { arm64: "arm64", x64: "x64" }[arch];
  if (!cpu) return null;
  if (platform === "darwin") return `darwin-${cpu}`;
  if (platform === "win32") return `win32-${cpu}-msvc`;
  if (platform === "linux") return `linux-${cpu}-${libc()}`;
  return null;
}

function resolveBinary() {
  if (process.env.PULSAR_BINARY) return process.env.PULSAR_BINARY;
  const suffix = platformSuffix();
  if (!suffix) {
    fail(`unsupported platform ${process.platform}/${process.arch}`);
  }
  const pkg = `${SCOPE}/${NAME}-${suffix}`;
  try {
    const manifest = require.resolve(`${pkg}/package.json`);
    const exe = process.platform === "win32" ? `${NAME}.exe` : NAME;
    return path.join(path.dirname(manifest), "bin", exe);
  } catch {
    fail(
      `the platform package ${pkg} is not installed.\n` +
        "Reinstall without --no-optional / --omit=optional, or set PULSAR_BINARY " +
        `to a ${NAME} executable.`,
    );
  }
}

function fail(message) {
  process.stderr.write(`${NAME}: ${message}\n`);
  process.exit(1);
}

const child = spawn(resolveBinary(), process.argv.slice(2), { stdio: "inherit" });
child.on("error", (err) => fail(`could not start the binary: ${err.message}`));
child.on("exit", (code, signal) => {
  if (signal) process.kill(process.pid, signal);
  else process.exit(code ?? 1);
});
for (const sig of ["SIGINT", "SIGTERM", "SIGHUP"]) {
  process.on(sig, () => child.kill(sig));
}
