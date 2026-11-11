#!/usr/bin/env node
// Prepare a release: set the version and date the changelog.
//
// The "Prepare release" workflow (release-pr.yml) runs this and opens the
// result as a release PR; merging it tags vX.Y.Z (release-tag.yml), and the
// tag runs release.yml (build every platform, publish the npm packages). By
// hand, from a clean main with tags:
//
//   node scripts/prepare-release.mjs --bump minor     # from the newest release
//   node scripts/prepare-release.mjs --version 1.0.0
//
// "Newest release" is the higher of the newest vX.Y.Z tag and the workspace
// version in Cargo.toml. Writes the version into Cargo.toml's
// [workspace.package] (every crate inherits it; release.yml checks the tag
// against it) and into Cargo.lock's entries for the workspace's own crates,
// stamps the Claude Code plugin (plugins/pulsar: plugin.json's version and
// the @krv-labs/pulsar@X.Y.Z that .mcp.json launches; reviewers reject an
// unpinned npx launcher), and renames CHANGELOG.md's [Unreleased] section to [X.Y.Z] - <today> under a
// fresh, empty [Unreleased]. Refuses an empty [Unreleased]: a release says
// what changed. Prints the version. No dependencies and no cargo, so it runs
// before any install.

import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";

const PLAIN = /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/;
const WORKSPACE_VERSION = /(\[workspace\.package\][^[]*?\nversion\s*=\s*")([^"]+)(")/;
const UNRELEASED = "## [Unreleased]";
const order = (v) => v.split(".").map(Number);
const newer = (a, b) => {
  const [x, y] = [order(a), order(b)];
  for (let i = 0; i < 3; i++) if (x[i] !== y[i]) return x[i] > y[i];
  return false;
};
const fail = (message) => {
  console.error(message);
  process.exit(1);
};

const args = process.argv.slice(2);
const flag = (name) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : undefined;
};
const bump = flag("--bump");
const exact = flag("--version")?.replace(/^v/, "");
if (!bump === !exact) fail("give exactly one of --bump patch|minor|major or --version X.Y.Z");

const cargo = readFileSync("Cargo.toml", "utf8");
const current = cargo.match(WORKSPACE_VERSION)?.[2];
if (!current || !PLAIN.test(current)) fail("no X.Y.Z [workspace.package] version in Cargo.toml");

const tags = execFileSync("git", ["tag", "--list", "v*"], { encoding: "utf8" })
  .split("\n")
  .map((tag) => tag.slice(1))
  .filter((v) => PLAIN.test(v));
const previous = [current, ...tags].reduce((a, b) => (newer(b, a) ? b : a));

let version = exact;
if (bump) {
  const [major, minor, patch] = order(previous);
  version =
    bump === "major" ? `${major + 1}.0.0`
    : bump === "minor" ? `${major}.${minor + 1}.0`
    : bump === "patch" ? `${major}.${minor}.${patch + 1}`
    : fail(`--bump must be patch, minor or major, not ${bump}`);
}
if (!PLAIN.test(version)) fail(`${version} is not X.Y.Z`);
if (!newer(version, previous)) fail(`${version} is not newer than the last release, ${previous}`);

const changelog = readFileSync("CHANGELOG.md", "utf8");
if (changelog.includes(`## [${version}]`)) fail(`CHANGELOG.md already has a [${version}] section`);
const start = changelog.indexOf(UNRELEASED);
if (start < 0) fail("CHANGELOG.md has no [Unreleased] section");
const bodyStart = start + UNRELEASED.length;
const next = changelog.indexOf("\n## [", bodyStart);
const notes = changelog.slice(bodyStart, next < 0 ? undefined : next);
// A `---` rule or a top-level heading ends the section too (the Python history below one).
if (!notes.split(/\n(?:---|# )/)[0].trim()) {
  fail("CHANGELOG.md's [Unreleased] section is empty: add the release notes first");
}
const today = new Date().toISOString().slice(0, 10);

// Workspace crates are the lockfile entries with no `source` (registry and git
// packages always have one). Each carries the workspace version.
const lock = readFileSync("Cargo.lock", "utf8");
let bumped = 0;
const newLock = lock
  .split("\n[[package]]\n")
  .map((block) => {
    if (/^source = /m.test(block) || !/^name = /m.test(block)) return block;
    const out = block.replace(/^version = "([^"]+)"$/m, (line, v) => {
      if (v !== current) fail(`Cargo.lock has a workspace crate at ${v}, not ${current}; run cargo check first`);
      bumped++;
      return `version = "${version}"`;
    });
    return out;
  })
  .join("\n[[package]]\n");
if (bumped === 0) fail("found no workspace crates in Cargo.lock");

// The plugin pins the package it launches; both must name the new version.
const PLUGIN = "plugins/pulsar/.claude-plugin/plugin.json";
const PLUGIN_MCP = "plugins/pulsar/.mcp.json";
const PIN = /@krv-labs\/pulsar@\d+\.\d+\.\d+/g;
const pluginText = readFileSync(PLUGIN, "utf8");
const pluginMcp = readFileSync(PLUGIN_MCP, "utf8");
if (!/^\s*"version":\s*"[^"]*"/m.test(pluginText)) fail(`${PLUGIN} has no version`);
if ((pluginMcp.match(PIN) || []).length !== 1) fail(`${PLUGIN_MCP} must launch exactly one pinned @krv-labs/pulsar@X.Y.Z`);

writeFileSync("Cargo.toml", cargo.replace(WORKSPACE_VERSION, `$1${version}$3`));
writeFileSync(PLUGIN, pluginText.replace(/^(\s*"version":\s*")[^"]*(")/m, `$1${version}$2`));
writeFileSync(PLUGIN_MCP, pluginMcp.replace(PIN, `@krv-labs/pulsar@${version}`));
writeFileSync("Cargo.lock", newLock);
writeFileSync(
  "CHANGELOG.md",
  changelog.slice(0, bodyStart) + `\n\n## [${version}] - ${today}` + changelog.slice(bodyStart),
);
console.log(version);
