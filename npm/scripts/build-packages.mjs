#!/usr/bin/env node
// Generate the npm packages for a release from built binaries:
//   @krv-labs/<NAME>               launcher + optionalDependencies
//   @krv-labs/<NAME>-<platform>    one prebuilt binary each (esbuild/biome pattern)
//
//   node npm/scripts/build-packages.mjs --binaries <dir> --out <dir>
//        [--version X.Y.Z] [--targets suffix,suffix] [--require-all]
//
// <dir>/<rust-target>/<NAME>[.exe] is expected for every platform built.
// The version defaults to the workspace version in Cargo.toml.
import { chmodSync, copyFileSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const SCOPE = "@krv-labs";
// The package, binary and command name (keep in step with the pulsar-cli
// [[bin]] name and pulsar_install::NAME).
export const NAME = "pulsar";
const REGISTRY = "https://registry.npmjs.org";
const REPOSITORY = "https://github.com/Krv-Labs/pulsar";

export const PLATFORMS = [
  { suffix: "darwin-arm64", target: "aarch64-apple-darwin", os: "darwin", cpu: "arm64" },
  { suffix: "darwin-x64", target: "x86_64-apple-darwin", os: "darwin", cpu: "x64" },
  { suffix: "linux-x64-gnu", target: "x86_64-unknown-linux-gnu", os: "linux", cpu: "x64", libc: "glibc" },
  { suffix: "linux-arm64-gnu", target: "aarch64-unknown-linux-gnu", os: "linux", cpu: "arm64", libc: "glibc" },
  { suffix: "linux-x64-musl", target: "x86_64-unknown-linux-musl", os: "linux", cpu: "x64", libc: "musl" },
  { suffix: "linux-arm64-musl", target: "aarch64-unknown-linux-musl", os: "linux", cpu: "arm64", libc: "musl" },
  { suffix: "win32-x64-msvc", target: "x86_64-pc-windows-msvc", os: "win32", cpu: "x64" },
  { suffix: "win32-arm64-msvc", target: "aarch64-pc-windows-msvc", os: "win32", cpu: "arm64" },
];

export function workspaceVersion() {
  const cargo = readFileSync(path.join(ROOT, "Cargo.toml"), "utf8");
  const m = cargo.match(/\[workspace\.package\][^[]*?\nversion\s*=\s*"([^"]+)"/);
  if (!m) throw new Error("no [workspace.package] version in Cargo.toml");
  return m[1];
}

const common = (version) => ({
  version,
  license: "BSD-3-Clause",
  repository: { type: "git", url: `git+${REPOSITORY}.git` },
  homepage: `${REPOSITORY}#readme`,
  bugs: { url: `${REPOSITORY}/issues` },
  publishConfig: { registry: REGISTRY, access: "public", provenance: true },
});

export function buildPackages({ binaries, out, version, targets, requireAll }) {
  const wanted = PLATFORMS.filter((p) => !targets || targets.includes(p.suffix));
  const built = [];
  for (const p of wanted) {
    const exe = p.os === "win32" ? `${NAME}.exe` : NAME;
    const src = path.join(binaries, p.target, exe);
    if (!existsSync(src)) {
      if (requireAll) throw new Error(`missing binary for ${p.suffix}: ${src}`);
      continue;
    }
    const dir = path.join(out, `${NAME}-${p.suffix}`);
    rmSync(dir, { recursive: true, force: true });
    mkdirSync(path.join(dir, "bin"), { recursive: true });
    copyFileSync(src, path.join(dir, "bin", exe));
    chmodSync(path.join(dir, "bin", exe), 0o755);
    const manifest = {
      name: `${SCOPE}/${NAME}-${p.suffix}`,
      description: `Pulsar binary for ${p.suffix}. Install ${SCOPE}/${NAME} instead.`,
      ...common(version),
      os: [p.os],
      cpu: [p.cpu],
      ...(p.libc ? { libc: [p.libc] } : {}),
      files: ["bin"],
    };
    writeFileSync(path.join(dir, "package.json"), JSON.stringify(manifest, null, 2) + "\n");
    built.push(p);
  }
  if (built.length === 0) throw new Error("no binaries found; nothing to package");

  const rootDir = path.join(out, NAME);
  rmSync(rootDir, { recursive: true, force: true });
  mkdirSync(path.join(rootDir, "bin"), { recursive: true });
  // The launcher template carries its own NAME constant; stamp ours into it.
  const launcher = readFileSync(path.join(ROOT, "npm/pulsar/bin/pulsar.js"), "utf8")
    .replace(/^const NAME = "[^"]*";$/m, `const NAME = "${NAME}";`);
  writeFileSync(path.join(rootDir, `bin/${NAME}.js`), launcher);
  chmodSync(path.join(rootDir, `bin/${NAME}.js`), 0o755);
  copyFileSync(path.join(ROOT, "npm/pulsar/README.md"), path.join(rootDir, "README.md"));
  const rootManifest = {
    name: `${SCOPE}/${NAME}`,
    description: "Pulsar: topological data analysis for tabular data, as an MCP server and CLI.",
    keywords: ["topological-data-analysis", "tda", "mcp", "mcp-server", "ball-mapper", "clustering"],
    ...common(version),
    bin: { [NAME]: `bin/${NAME}.js` },
    files: ["bin", "README.md"],
    engines: { node: ">=18" },
    optionalDependencies: Object.fromEntries(
      built.map((p) => [`${SCOPE}/${NAME}-${p.suffix}`, version]),
    ),
  };
  writeFileSync(path.join(rootDir, "package.json"), JSON.stringify(rootManifest, null, 2) + "\n");
  return { root: rootDir, platforms: built.map((p) => path.join(out, `${NAME}-${p.suffix}`)), built };
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const { values } = parseArgs({
    options: {
      binaries: { type: "string" },
      out: { type: "string" },
      version: { type: "string" },
      targets: { type: "string" },
      "require-all": { type: "boolean", default: false },
    },
  });
  if (!values.binaries || !values.out) {
    console.error("usage: build-packages.mjs --binaries <dir> --out <dir> [--version X] [--targets a,b] [--require-all]");
    process.exit(2);
  }
  const res = buildPackages({
    binaries: path.resolve(values.binaries),
    out: path.resolve(values.out),
    version: values.version ?? workspaceVersion(),
    targets: values.targets?.split(","),
    requireAll: values["require-all"],
  });
  console.log(`built ${res.built.length} platform package(s) + launcher in ${values.out}`);
}
