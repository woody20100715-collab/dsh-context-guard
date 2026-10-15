// Tool-output compression is lossy, so originals are kept here and the model can pull them
// back by id with context_retrieve instead of re-running a side-effecting tool.

import { mkdir, readFile, unlink, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { resolve, sep } from 'node:path';
import { randomUUID } from 'node:crypto';
import type { ResolvedConfig } from './config.ts';
import { expandHome } from './config.ts';
import { GuardLogger } from './utils/logger.ts';

/** `ccr_` + 16 hex chars: naming convention and the first line of defence against path traversal. */
export const CCR_ID_PATTERN = /^ccr_[0-9a-f]{16}$/;

export interface CCRRecord {
  readonly id: string;
  readonly text: string;
  readonly createdAt: number;
  /** LRU clock. */
  lastAccess: number;
  /** UTF-8 byte length. */
  readonly bytes: number;
  /** Source tool, for context when the content is retrieved. */
  readonly toolName?: string;
  /** Token estimate of the original text. */
  readonly tokens?: number;
  persisted: boolean;
}

export interface CCRStats {
  readonly entries: number;
  readonly bytes: number;
  /** Empty when persistence is disabled. */
  readonly path: string;
  readonly persist: boolean;
  /** Cumulative LRU + TTL evictions. */
  readonly evicted: number;
  /** Entries kept in memory only because they exceed ccrMaxChars. */
  readonly skippedPersist: number;
}

/** On-disk shape; deliberately not CCRRecord, so lastAccess is never rewritten to disk. */
interface CCRFilePayload {
  readonly id: string;
  readonly text: string;
  readonly createdAt: number;
  readonly toolName?: string;
  readonly tokens?: number;
}

export interface RetrievalSlice {
  readonly content: string;
  /** Clamped to the content length. */
  readonly offset: number;
  readonly totalChars: number;
  readonly truncated: boolean;
}

// Retrieval is never re-compressed - the model would never reach the full original otherwise -
// so paging is the only thing that keeps a single retrieve from blowing the context window.
export function sliceForRetrieval(text: string, offset: number, limit: number): RetrievalSlice {
  const totalChars = text.length;
  const size = Math.max(1, Math.floor(limit));
  const start = Math.max(0, Math.min(Math.floor(offset), Math.max(0, totalChars - 1)));
  const content = text.slice(start, start + size);
  return { content, offset: start, totalChars, truncated: start + content.length < totalChars };
}

/** In-memory originals plus optional disk persistence, with LRU/TTL eviction. */
export class CCRStore {
  private readonly records = new Map<string, CCRRecord>();
  private readonly config: ResolvedConfig;
  private readonly logger: GuardLogger;
  private readonly rootDir: string;
  private totalBytes = 0;
  private evicted = 0;
  private skippedPersist = 0;
  private ready: Promise<void> | undefined;

  /** `home` is overridable for tests. */
  constructor(config: ResolvedConfig, logger: GuardLogger, home: string = homedir()) {
    this.config = config;
    this.logger = logger;
    this.rootDir = config.ccrPersist ? resolve(expandHome(config.ccrStorePath, home)) : '';
  }

  static newId(): string {
    return `ccr_${randomUUID().replaceAll('-', '').slice(0, 16)}`;
  }

  /** Invalid ids count as missing and never reach the filesystem. */
  static isValidId(id: string): boolean {
    return CCR_ID_PATTERN.test(id);
  }

  private ensureReady(): Promise<void> {
    if (!this.config.ccrPersist) return Promise.resolve();
    this.ready ??= mkdir(this.rootDir, { recursive: true, mode: 0o700 })
      .then(() => undefined)
      .catch((error: unknown) => {
        this.logger.warn(`failed to create the CCR store directory, falling back to memory only: ${GuardLogger.describe(error)}`);
      });
    return this.ready;
  }

  /** Defence in depth: callers already ran isValidId, we still assert the result stays under rootDir. */
  private filePathFor(id: string): string {
    const target = resolve(this.rootDir, `${id}.json`);
    if (!target.startsWith(this.rootDir + sep)) {
      throw new Error(`CCR path traversal rejected for id ${id}`);
    }
    return target;
  }

  /** Returns the retrieval id, or undefined when CCR is off or the entry is rejected. */
  store(text: string, meta?: { toolName?: string; tokens?: number }): string | undefined {
    if (!this.config.enableCCR || this.config.ccrMaxEntries <= 0) return undefined;
    if (text.length === 0) return undefined;

    const id = CCRStore.newId();
    const bytes = Buffer.byteLength(text, 'utf8');
    const record: CCRRecord = {
      id,
      text,
      createdAt: Date.now(),
      lastAccess: Date.now(),
      bytes,
      ...(meta?.toolName === undefined ? {} : { toolName: meta.toolName }),
      ...(meta?.tokens === undefined ? {} : { tokens: meta.tokens }),
      persisted: false
    };
    this.records.set(id, record);
    this.totalBytes += bytes;
    this.prune();

    // oversized entries stay in memory only, so one huge output cannot fill the disk.
    if (this.config.ccrPersist && text.length <= this.config.ccrMaxChars) {
      void this.persist(record);
    } else if (this.config.ccrPersist) {
      this.skippedPersist += 1;
      this.logger.debug(`CCR entry ${id} exceeds ccrMaxChars (${this.config.ccrMaxChars}), keeping it in memory only`);
    }
    return id;
  }

  /** Fire and forget: a failed write only warns, it never breaks compression. */
  private async persist(record: CCRRecord): Promise<void> {
    try {
      await this.ensureReady();
      const payload: CCRFilePayload = {
        id: record.id,
        text: record.text,
        createdAt: record.createdAt,
        ...(record.toolName === undefined ? {} : { toolName: record.toolName }),
        ...(record.tokens === undefined ? {} : { tokens: record.tokens })
      };
      // 0600: tool output can contain credentials or internal addresses.
      await writeFile(this.filePathFor(record.id), JSON.stringify(payload), { encoding: 'utf8', mode: 0o600 });
      record.persisted = true;
    } catch (error) {
      this.logger.debug(`CCR persist failed (entry kept in memory): ${GuardLogger.describe(error)}`);
    }
  }

  /** Returns null for a missing, expired or malformed id. */
  retrieve(id: string): CCRRecord | null {
    if (!CCRStore.isValidId(id)) return null;
    const hit = this.records.get(id);
    if (hit !== undefined) {
      if (this.isExpired(hit)) {
        this.delete(id);
        return null;
      }
      hit.lastAccess = Date.now();
      // Map keeps insertion order: delete + set moves the entry to the end (LRU).
      this.records.delete(id);
      this.records.set(id, hit);
      return hit;
    }
    return null;
  }

  /** Disk fallback, so entries survive a process restart. */
  async retrieveFromDisk(id: string): Promise<CCRRecord | null> {
    const memory = this.retrieve(id);
    if (memory !== null) return memory;
    if (!this.config.ccrPersist || !CCRStore.isValidId(id)) return null;
    try {
      const raw = await readFile(this.filePathFor(id), 'utf8');
      const payload = JSON.parse(raw) as CCRFilePayload;
      if (typeof payload.text !== 'string') return null;
      const record: CCRRecord = {
        id,
        text: payload.text,
        createdAt: typeof payload.createdAt === 'number' ? payload.createdAt : Date.now(),
        lastAccess: Date.now(),
        bytes: Buffer.byteLength(payload.text, 'utf8'),
        ...(typeof payload.toolName === 'string' ? { toolName: payload.toolName } : {}),
        ...(typeof payload.tokens === 'number' ? { tokens: payload.tokens } : {}),
        persisted: true
      };
      if (this.isExpired(record)) {
        void this.removeFile(id);
        return null;
      }
      // a disk hit re-enters the memory cache and may be evicted right away.
      this.records.set(id, record);
      this.totalBytes += record.bytes;
      this.prune();
      return record;
    } catch {
      return null;
    }
  }

  private isExpired(record: CCRRecord): boolean {
    return this.config.ccrTtlMs > 0 && Date.now() - record.createdAt > this.config.ccrTtlMs;
  }

  /** Returns whether an entry was actually removed. */
  private delete(id: string): boolean {
    const record = this.records.get(id);
    if (record === undefined) return false;
    this.records.delete(id);
    this.totalBytes -= record.bytes;
    return true;
  }

  private async removeFile(id: string): Promise<void> {
    try {
      await unlink(this.filePathFor(id));
    } catch {
      // best effort: a missing file or a permission error is not worth reporting.
    }
  }

  /** TTL first, then LRU eviction until both the entry and byte caps hold. */
  prune(): number {
    let removed = 0;
    if (this.config.ccrTtlMs > 0) {
      const now = Date.now();
      for (const [id, record] of [...this.records]) {
        if (now - record.createdAt > this.config.ccrTtlMs) {
          this.delete(id);
          void this.removeFile(id);
          removed += 1;
        }
      }
    }
    // from the oldest (least recently used) entry onwards.
    const overEntries = (): boolean => this.records.size > this.config.ccrMaxEntries;
    const overBytes = (): boolean => this.totalBytes > this.config.ccrMaxBytes;
    while ((overEntries() || overBytes()) && this.records.size > 0) {
      const oldest = this.records.keys().next();
      if (oldest.done) break;
      const record = this.records.get(oldest.value);
      this.delete(oldest.value);
      if (record?.persisted === true) void this.removeFile(oldest.value);
      removed += 1;
    }
    this.evicted += removed;
    return removed;
  }

  /** Used by tests and hot reload. */
  clear(): void {
    this.records.clear();
    this.totalBytes = 0;
  }

  stats(): CCRStats {
    return {
      entries: this.records.size,
      bytes: this.totalBytes,
      path: this.rootDir,
      persist: this.config.ccrPersist,
      evicted: this.evicted,
      skippedPersist: this.skippedPersist
    };
  }

  /** LRU order, oldest first. */
  ids(): string[] {
    return [...this.records.keys()];
  }

  get directory(): string {
    return this.rootDir;
  }
}
