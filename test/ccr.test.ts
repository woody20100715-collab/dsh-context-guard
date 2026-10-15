import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm, readdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { resolveConfig } from '../src/config.ts';
import { CCRStore, sliceForRetrieval } from '../src/ccr.ts';
import { GuardLogger, MemorySink } from '../src/utils/logger.ts';

const logger = new GuardLogger('error', new MemorySink());

async function makeDir(): Promise<string> {
  return mkdtemp(join(tmpdir(), 'cg-ccr-'));
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

test('ccr: store/retrieve round-trips and ids look like ccr_<16 hex>', async () => {
  const dir = await makeDir();
  try {
    const store = new CCRStore(resolveConfig({ ccrStorePath: dir, ccrPersist: false }), logger);
    const id = store.store('hello world', { toolName: 'demo', tokens: 3 });
    assert.ok(id);
    assert.match(id, /^ccr_[0-9a-f]{16}$/);
    const record = store.retrieve(id);
    assert.equal(record?.text, 'hello world');
    assert.equal(record?.toolName, 'demo');
    assert.equal(record?.tokens, 3);
    assert.equal(store.stats().entries, 1);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('ccr: invalid ids are never found (path-traversal guard)', async () => {
  const dir = await makeDir();
  try {
    const store = new CCRStore(resolveConfig({ ccrStorePath: dir }), logger);
    for (const bad of ['../../etc/passwd', 'ccr_', 'ccr_XYZ', 'CCR_0123456789abcdef', 'ccr_0123456789abcde', 'ccr_0123456789abcdef0']) {
      assert.equal(CCRStore.isValidId(bad), false, `id must be rejected: ${bad}`);
      assert.equal(store.retrieve(bad), null);
      assert.equal(await store.retrieveFromDisk(bad), null);
    }
    assert.equal(CCRStore.isValidId('ccr_0123456789abcdef'), true);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('ccr: over the entry cap it evicts the least recently used', async () => {
  const dir = await makeDir();
  try {
    const store = new CCRStore(resolveConfig({ ccrStorePath: dir, ccrPersist: false, ccrMaxEntries: 3 }), logger);
    const a = store.store('A');
    const b = store.store('B');
    const c = store.store('C');
    assert.ok(a && b && c);
    // touch a so it becomes most-recently-used and b becomes the LRU victim
    assert.equal(store.retrieve(a)?.text, 'A');
    const d = store.store('D');
    assert.ok(d);
    assert.equal(store.stats().entries, 3);
    assert.equal(store.retrieve(b), null, 'b should be evicted');
    assert.equal(store.retrieve(a)?.text, 'A');
    assert.equal(store.retrieve(c)?.text, 'C');
    assert.equal(store.retrieve(d)?.text, 'D');
    assert.ok(store.stats().evicted >= 1);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('ccr: over the byte cap it evicts until everything fits', async () => {
  const dir = await makeDir();
  try {
    const store = new CCRStore(resolveConfig({ ccrStorePath: dir, ccrPersist: false, ccrMaxBytes: 100, ccrMaxEntries: 100 }), logger);
    const first = store.store('x'.repeat(60));
    const second = store.store('y'.repeat(60));
    assert.ok(first && second);
    assert.equal(store.retrieve(first), null, 'oldest entry should be evicted to make room');
    assert.equal(store.retrieve(second)?.text.length, 60);
    assert.ok(store.stats().bytes <= 100);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('ccr: entries are gone past their TTL', async () => {
  const dir = await makeDir();
  try {
    const store = new CCRStore(resolveConfig({ ccrStorePath: dir, ccrPersist: false, ccrTtlMs: 30 }), logger);
    const id = store.store('expiring');
    assert.ok(id);
    assert.equal(store.retrieve(id)?.text, 'expiring');
    await sleep(60);
    assert.equal(store.retrieve(id), null);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('ccr: a new instance reads back what was persisted (process restart)', async () => {
  const dir = await makeDir();
  try {
    const config = resolveConfig({ ccrStorePath: dir, ccrPersist: true, ccrTtlMs: 60000 });
    const first = new CCRStore(config, logger);
    const id = first.store('persisted payload', { toolName: 'pwsh', tokens: 4 });
    assert.ok(id);
    await sleep(150); // let the async write land

    const files = await readdir(dir);
    assert.deepEqual(files, [`${id}.json`]);

    const second = new CCRStore(config, logger);
    const record = await second.retrieveFromDisk(id);
    assert.equal(record?.text, 'persisted payload');
    assert.equal(record?.toolName, 'pwsh');
    // read-back warms the in-memory cache
    assert.equal(second.retrieve(id)?.text, 'persisted payload');
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('ccr: entries over ccrMaxChars stay in memory only', async () => {
  const dir = await makeDir();
  try {
    const store = new CCRStore(resolveConfig({ ccrStorePath: dir, ccrPersist: true, ccrMaxChars: 1024 }), logger);
    const id = store.store('z'.repeat(2048));
    assert.ok(id);
    await sleep(100);
    assert.deepEqual(await readdir(dir), []);
    assert.equal(store.retrieve(id)?.text.length, 2048);
    assert.equal(store.stats().skippedPersist, 1);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test('ccr: paging returns exact offsets and a has-more flag', () => {
  const text = 'abcdefghij';
  assert.deepEqual(sliceForRetrieval(text, 0, 4), { content: 'abcd', offset: 0, totalChars: 10, truncated: true });
  assert.deepEqual(sliceForRetrieval(text, 4, 4), { content: 'efgh', offset: 4, totalChars: 10, truncated: true });
  assert.deepEqual(sliceForRetrieval(text, 8, 4), { content: 'ij', offset: 8, totalChars: 10, truncated: false });
  // out-of-range offset clamps to the last page; a bad limit still yields one char
  assert.deepEqual(sliceForRetrieval(text, 999, 4), { content: 'j', offset: 9, totalChars: 10, truncated: false });
  assert.deepEqual(sliceForRetrieval(text, -5, 3), { content: 'abc', offset: 0, totalChars: 10, truncated: true });
  assert.equal(sliceForRetrieval(text, 0, 0).content.length, 1);
  // empty content must not throw
  assert.deepEqual(sliceForRetrieval('', 0, 10), { content: '', offset: 0, totalChars: 0, truncated: false });
});

test('ccr: nothing is stored when disabled', async () => {
  const dir = await makeDir();
  try {
    const store = new CCRStore(resolveConfig({ ccrStorePath: dir, enableCCR: false }), logger);
    assert.equal(store.store('nope'), undefined);
    assert.equal(store.stats().entries, 0);
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
