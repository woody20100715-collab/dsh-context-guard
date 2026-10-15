import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { parse } from 'yaml';

import { resolveConfig, DEFAULTS } from '../src/config.ts';

const here = dirname(fileURLToPath(import.meta.url));
const patchPath = join(here, '..', 'cordis.patch.yml');

// patch entry, partial: only the fields this test reads
interface PatchEntry {
  readonly id?: string;
  readonly name?: string;
  readonly disabled?: boolean;
  readonly config?: Record<string, unknown>;
  readonly insert?: readonly PatchEntry[];
}

test('patch: top level is an array that both disables the built-in backend and mounts this plugin', async () => {
  const entries = parse(await readFile(patchPath, 'utf8')) as PatchEntry[];
  assert.ok(Array.isArray(entries));

  const disabled = entries.find((entry) => entry.id === 'compaction-basic');
  assert.ok(disabled, 'compaction-basic must be disabled explicitly: ctx.compaction is a singleton and two backends would collide');
  assert.equal(disabled.disabled, true);

  const insert = entries.find((entry) => Array.isArray(entry.insert));
  assert.ok(insert?.insert, 'new plugins must be mounted via the insert list; top-level entries override by id');
  const row = insert.insert.find((entry) => entry.id === 'context-guard');
  assert.ok(row, 'there must be a row with id context-guard');
  assert.equal(row.name, 'dsh-context-guard');
});

test('patch: the sample config uses only known keys and passes semantic validation', async () => {
  const entries = parse(await readFile(patchPath, 'utf8')) as PatchEntry[];
  const row = entries.find((entry) => Array.isArray(entry.insert))?.insert?.find((entry) => entry.id === 'context-guard');
  assert.ok(row?.config);

  const resolved = resolveConfig(row.config);
  assert.deepEqual([...resolved.unknownKeys], [], 'no unknown keys in the sample config: users copy it verbatim');
  assert.equal(resolved.summaryLanguage, 'auto');
  assert.equal(resolved.thresholdRatio, 0.8);
  assert.ok(resolved.retainRatio < resolved.thresholdRatio);
  assert.ok(resolved.targetToolOutputTokens <= resolved.maxToolOutputTokens);
  assert.equal(resolved.minToolOutputTokens <= resolved.maxToolOutputTokens, true);
});

test('patch: every configured key exists in DEFAULTS, so docs cannot drift from code', async () => {
  const entries = parse(await readFile(patchPath, 'utf8')) as PatchEntry[];
  const config = entries.find((entry) => Array.isArray(entry.insert))?.insert?.find((entry) => entry.id === 'context-guard')?.config ?? {};
  const defaults = new Set(Object.keys(DEFAULTS));
  for (const key of Object.keys(config)) {
    assert.ok(defaults.has(key), `config key ${key} is missing from DEFAULTS`);
  }
});
