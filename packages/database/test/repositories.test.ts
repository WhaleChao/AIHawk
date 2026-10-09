import { APPROVAL_LIST_LIMIT, newId, parseDotConfig, TASK_LIST_LIMIT, vmName, type OutboundEvent } from "@invisible-dots/shared";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { DotChangedError, DotNameTakenError, TransactionMisuseError, type Database, type Repositories } from "../src/index.js";
import { EventsRepository } from "../src/events.js";
import { loadMigrations } from "../src/migrate.js";
import { createTestDatabase, testAdapters, type TestDatabase } from "../src/testing.js";

const SETUP_TIMEOUT = 60_000;

/** The rows of `table` with these ids, created one second apart in this order: two inserts in a row can get the same clock reading, and the order is then the ids', which are random. */
async function createdInOrder(db: Database, table: "tasks" | "approvals", ids: readonly string[]) {
  await db.query(
    `UPDATE ${table} t SET created_at = TIMESTAMPTZ '2026-01-01T00:00:00Z' + o.n * INTERVAL '1 second' FROM unnest($1::text[]) WITH ORDINALITY AS o(id, n) WHERE t.id = o.id`,
    [[...ids]],
  );
}

const yaml = (name: string) => `name: ${name}\nmodel:\n  provider: openrouter\n  id: test/model\n`;

async function seedDot(r: Repositories, name: string) {
  const id = newId("dot");
  const dot = await r.dots.insert({ id, config: parseDotConfig(yaml(name)), status: "READY" });
  await r.computers.insert({ dotId: id, vmName: vmName(id), state: "RUNNING", token: `tok-${name}` });
  return dot;
}

function outbound(seq: number, type: OutboundEvent["type"], data: Record<string, unknown>): OutboundEvent {
  return { seq, id: `evt-${seq}`, type, ts: new Date().toISOString(), data } as OutboundEvent;
}

describe.each(testAdapters())("repositories on %s", { timeout: SETUP_TIMEOUT }, (kind) => {
  let t: TestDatabase;
  let db: Database;

  beforeAll(async () => {
    t = await createTestDatabase(kind);
    db = t.db;
  }, SETUP_TIMEOUT);

  afterAll(async () => {
    await t?.drop();
  }, SETUP_TIMEOUT);

  it("migrate is idempotent and records each version", async () => {
    expect(db.kind).toBe(kind);
    const again = await db.migrate();
    expect(again.applied).toEqual([]);
    expect(again.alreadyApplied).toContain("0001_initial");
    const { rows } = await db.query<{ version: string }>("SELECT version FROM schema_migrations ORDER BY version");
    expect(rows.map((r) => r.version)).toEqual([
      "0001_initial",
      "0002_inbound_events",
      "0003_task_spend",
      "0004_channels",
      "0005_channel_prompts",
      "0006_events_task",
      "0007_dot_config_version",
      "0008_computer_next_automation",
      "0009_dot_keeps_its_memory",
      "0010_dot_has_no_goal",
      "0011_dot_has_no_browser_settings",
      "0012_dot_has_no_context_tokens",
    ]);
  });

  it("0012: a config saved with limits.context_tokens loses it and moves its version; the other limits stay; one without is left as it was", async () => {
    const old = await seedDot(db, "alpha-context");
    const untouched = await seedDot(db, "alpha-context-none");
    await db.query(`UPDATE dots SET config = jsonb_set(config, '{limits,context_tokens}', '32000') WHERE id = $1`, [old.id]);
    const sql = (await loadMigrations()).find((migration) => migration.version === "0012_dot_has_no_context_tokens")!.sql;

    for (const statement of sql.split(/;\s*\n/).filter((part) => part.trim() !== "")) await db.query(statement);

    const after = await db.dots.get(old.id);
    expect("context_tokens" in after!.config.limits).toBe(false);
    expect(after?.config).toEqual(old.config);
    expect(after?.config_version).toBe(old.config_version + 1);
    expect((await db.dots.get(untouched.id))?.config_version).toBe(untouched.config_version);
  });

  it("0011: a config saved with browser settings loses them and moves its version; one without is left as it was", async () => {
    const old = await seedDot(db, "alpha-browser");
    const untouched = await seedDot(db, "alpha-browser-none");
    await db.query(
      `UPDATE dots SET config = config || '{"browser":{"identities":{"managed_by_dot":false,"max_identities":5,"max_open":2}}}'::jsonb WHERE id = $1`,
      [old.id],
    );
    const sql = (await loadMigrations()).find((migration) => migration.version === "0011_dot_has_no_browser_settings")!.sql;

    for (const statement of sql.split(/;\s*\n/).filter((part) => part.trim() !== "")) await db.query(statement);

    const after = await db.dots.get(old.id);
    expect("browser" in after!.config).toBe(false);
    expect(after?.config).toEqual(old.config);
    expect(after?.config_version).toBe(old.config_version + 1);
    expect((await db.dots.get(untouched.id))?.config_version).toBe(untouched.config_version);
  });

  it("0010: a config saved with a goal loses it and moves its version; one without is left as it was", async () => {
    const old = await seedDot(db, "alpha-goal");
    const untouched = await seedDot(db, "alpha-goal-none");
    await db.query(`UPDATE dots SET config = config || '{"goal":"watch the fares"}'::jsonb WHERE id = $1`, [old.id]);
    const sql = (await loadMigrations()).find((migration) => migration.version === "0010_dot_has_no_goal")!.sql;

    for (const statement of sql.split(/;\s*\n/).filter((part) => part.trim() !== "")) await db.query(statement);

    const after = await db.dots.get(old.id);
    expect("goal" in after!.config).toBe(false);
    expect(after?.config).toEqual(old.config);
    expect(after?.config_version).toBe(old.config_version + 1);
    expect((await db.dots.get(untouched.id))?.config_version).toBe(untouched.config_version);
  });

  it("0009: a config saved with the memory switch and permission loses both and moves its version, and memory.written rows go", async () => {
    const old = await seedDot(db, "alpha-memory");
    const untouched = await seedDot(db, "alpha-memory-none");
    await db.query(
      `UPDATE dots SET config = jsonb_set(config || '{"memory":{"enabled":false}}'::jsonb, '{permissions}', '{"memory.read":"allow","files.read":"ask"}'::jsonb) WHERE id = $1`,
      [old.id],
    );
    await db.query(
      `INSERT INTO events (dot_id, type, data, source) VALUES ($1, 'memory.written', '{"key":"a.md"}'::jsonb, 'host'), ($1, 'agent.state', '{"state":"IDLE"}'::jsonb, 'host')`,
      [old.id],
    );
    const sql = (await loadMigrations()).find((migration) => migration.version === "0009_dot_keeps_its_memory")!.sql;

    // Its statements one at a time: the repositories run single statements; `migrate` runs the file as a script.
    for (const statement of sql.split(/;\s*\n/).filter((part) => part.trim() !== "")) await db.query(statement);

    const after = await db.dots.get(old.id);
    expect(after?.config).toEqual({ ...old.config, permissions: { "files.read": "ask" } });
    expect(after?.config_version).toBe(old.config_version + 1);
    expect((await db.dots.get(untouched.id))?.config_version).toBe(untouched.config_version);
    const { rows } = await db.query<{ type: string }>("SELECT type FROM events WHERE dot_id = $1 AND type IN ('memory.written', 'agent.state')", [old.id]);
    expect(rows.map((row) => row.type)).toEqual(["agent.state"]);
  });

  it("dots: unique names, resolve by id or name, status with error", async () => {
    const dot = await seedDot(db, "alpha");
    await expect(
      db.dots.insert({ id: newId("dot"), config: parseDotConfig(yaml("alpha")), status: "CREATING" }),
    ).rejects.toBeInstanceOf(DotNameTakenError);
    expect((await db.dots.resolve("alpha"))?.id).toBe(dot.id);
    expect((await db.dots.resolve(dot.id))?.computer_state).toBe("RUNNING");
    expect((await db.dots.get(dot.id))?.config.name).toBe("alpha");
    const errored = await db.dots.setStatus(dot.id, "ERROR", "boom");
    expect(errored?.error).toBe("boom");
    expect((await db.dots.setStatus(dot.id, "READY"))?.error).toBeNull();
    expect((await db.dots.list()).map((d) => d.name)).toContain("alpha");
  });

  it("dots: setPermission changes one permission and nothing else, also when the config has none", async () => {
    const dot = await seedDot(db, "alpha-permissions");
    const before = dot.config;
    const first = await db.dots.setPermission(dot.id, "computer.exec", "allow");
    expect(first?.config).toEqual({ ...before, permissions: { "computer.exec": "allow" } });
    const second = await db.dots.setPermission(dot.id, "files.write", "ask");
    expect(second?.config.permissions).toEqual({ "computer.exec": "allow", "files.write": "ask" });
    expect((await db.dots.setPermission(dot.id, "computer.exec", "deny"))?.config.permissions).toEqual({ "computer.exec": "deny", "files.write": "ask" });
    expect((await db.dots.get(dot.id))?.config).toEqual({ ...before, permissions: { "computer.exec": "deny", "files.write": "ask" } });
    expect(Date.parse(second!.updated_at)).toBeGreaterThanOrEqual(Date.parse(first!.updated_at));

    await db.query("UPDATE dots SET config = config - 'permissions' WHERE id = $1", [dot.id]);
    expect((await db.dots.setPermission(dot.id, "automations", "allow"))?.config.permissions).toEqual({ automations: "allow" });
    expect(await db.dots.setPermission("dot_missing", "automations", "allow")).toBeNull();
  });

  it("dots: updateConfig with the config_version that was read saves once; an older one is DotChangedError and writes nothing", async () => {
    const dot = await seedDot(db, "alpha-stale");
    expect(dot.config_version).toBe(1);
    const next = { ...dot.config, instructions: "first" };
    const saved = await db.dots.updateConfig(dot.id, next, dot.config_version);
    expect(saved).toMatchObject({ config_version: 2 });
    expect(saved?.config.instructions).toBe("first");
    await expect(db.dots.updateConfig(dot.id, { ...dot.config, instructions: "second" }, dot.config_version)).rejects.toBeInstanceOf(DotChangedError);
    // An "always allow" is a save of the config too.
    const allowed = await db.dots.setPermission(dot.id, "computer.exec", "allow");
    expect(allowed?.config_version).toBe(3);
    await expect(db.dots.updateConfig(dot.id, { ...dot.config, instructions: "third" }, saved!.config_version)).rejects.toBeInstanceOf(DotChangedError);
    expect((await db.dots.get(dot.id))?.config).toEqual({ ...next, permissions: { ...dot.config.permissions, "computer.exec": "allow" } });
    expect(await db.dots.updateConfig("dot_missing", dot.config, 1)).toBeNull();
    // Without a precondition it replaces, as it always did.
    expect((await db.dots.updateConfig(dot.id, { ...dot.config, instructions: "plain" }))?.config.instructions).toBe("plain");
  });

  it("dots: a status change moves updated_at but not config_version, so it never makes a save from an old read stale", async () => {
    const dot = await seedDot(db, "alpha-status");
    const running = await db.dots.setStatus(dot.id, "RUNNING");
    expect(running).toMatchObject({ status: "RUNNING", config_version: dot.config_version });
    expect(Date.parse(running!.updated_at)).toBeGreaterThanOrEqual(Date.parse(dot.updated_at));
    const saved = await db.dots.updateConfig(dot.id, { ...dot.config, instructions: "after a status change" }, dot.config_version);
    expect(saved).toMatchObject({ config_version: dot.config_version + 1, status: "RUNNING" });
  });

  it("computers: token stored encrypted, process recorded and cleared, cursor only moves forward", async () => {
    const dot = await seedDot(db, "bravo");
    const { rows } = await db.query<{ token_enc: Uint8Array }>("SELECT token_enc FROM computers WHERE dot_id = $1", [dot.id]);
    expect(Buffer.from(rows[0]!.token_enc).includes(Buffer.from("tok-bravo"))).toBe(false);
    expect(await db.computers.token(dot.id)).toBe("tok-bravo");

    const fresh = await db.computers.get(dot.id);
    expect(fresh).toMatchObject({ vm_name: `invisible-dot-${dot.id}`, guest_port: null, pid: null, event_cursor: 0 });
    expect(await db.computers.setProcess(dot.id, { pid: 4242, guestPort: 40123 })).toMatchObject({
      pid: 4242,
      guest_port: 40123,
    });
    expect(await db.computers.setProcess(dot.id, null)).toMatchObject({ pid: null, guest_port: null });
    await expect(db.computers.setProcess(dot.id, { pid: 1, guestPort: 70000 })).rejects.toMatchObject({ code: "23514" });

    await db.computers.advanceCursor(dot.id, 7, new Date("2030-01-01T00:00:00Z"));
    await db.computers.advanceCursor(dot.id, 3, new Date("2030-01-01T00:00:01Z"));
    const computer = await db.computers.get(dot.id);
    expect(computer?.event_cursor).toBe(7);
    expect(computer?.last_active_at).toBe("2030-01-01T00:00:01.000Z");
    expect((await db.computers.setState(dot.id, "ERROR", "qemu failed"))?.last_error).toBe("qemu failed");
    expect((await db.computers.setState(dot.id, "STOPPED"))?.last_error).toBe("qemu failed");
    expect((await db.computers.setState(dot.id, "RUNNING", null))?.last_error).toBeNull();
    await db.computers.setImages(dot.id, "golden-1", "runtime-1");
    expect(await db.computers.get(dot.id)).toMatchObject({ golden_image: "golden-1", runtime_image: "runtime-1" });
  });

  it("computers: the next automation time starts empty, is kept as the guest reports it and is cleared by null", async () => {
    const dot = await seedDot(db, "next-automation");
    expect((await db.computers.get(dot.id))?.next_automation_at).toBeNull();

    await db.computers.setNextAutomation(dot.id, Date.parse("2030-03-04T05:06:07.000Z"));
    expect((await db.computers.get(dot.id))?.next_automation_at).toBe("2030-03-04T05:06:07.000Z");
    const before = (await db.computers.get(dot.id))!.updated_at;
    await db.computers.setNextAutomation(dot.id, Date.parse("2030-03-05T00:00:00.000Z"));
    expect((await db.computers.list()).find((c) => c.dot_id === dot.id)?.next_automation_at).toBe("2030-03-05T00:00:00.000Z");
    // It is the guest's report, not activity of the computer: nothing else of the row moves.
    expect((await db.computers.get(dot.id))?.updated_at).toBe(before);

    await db.computers.setNextAutomation(dot.id, null);
    expect((await db.computers.get(dot.id))?.next_automation_at).toBeNull();
  });

  it("computers: stop_reason is kept through STOPPING and STOPPED and cleared by any other state", async () => {
    const dot = await seedDot(db, "stop-reason");
    expect((await db.computers.get(dot.id))?.stop_reason).toBeNull();

    // A reason given with STOPPING stays for the STOPPED that follows (the interrupted stop and its recovery rely on it).
    expect((await db.computers.setState(dot.id, "STOPPING", undefined, "idle"))?.stop_reason).toBe("idle");
    expect((await db.computers.setState(dot.id, "STOPPED"))?.stop_reason).toBe("idle");
    // The person's stop of a computer that is already off replaces it.
    expect((await db.computers.setState(dot.id, "STOPPED", undefined, "user"))?.stop_reason).toBe("user");
    expect((await db.computers.setState(dot.id, "STOPPED", null))?.stop_reason).toBe("user");
    // Running again, in any way, clears it.
    expect((await db.computers.setState(dot.id, "STARTING"))?.stop_reason).toBeNull();
    expect((await db.computers.setState(dot.id, "STOPPED", null, "exited"))?.stop_reason).toBe("exited");
    expect((await db.computers.setState(dot.id, "ERROR", "boom"))?.stop_reason).toBeNull();
    await expect(db.query("UPDATE computers SET stop_reason = 'bored' WHERE dot_id = $1", [dot.id])).rejects.toThrow();
  });

  it("computers: stoppedWithAutomationBy leaves out the computers the person stopped, until they run again", async () => {
    const dot = await seedDot(db, "stopped-by-person");
    await db.computers.setNextAutomation(dot.id, Date.parse("2030-01-01T10:00:00Z"));
    const by = new Date("2030-01-01T10:30:00Z");
    const listed = async () => (await db.computers.stoppedWithAutomationBy(by)).includes(dot.id);

    await db.computers.setState(dot.id, "STOPPED", null, "user");
    expect(await listed()).toBe(false);
    await db.computers.setState(dot.id, "STOPPED", null, "idle");
    expect(await listed()).toBe(true);
    await db.computers.setState(dot.id, "STOPPED", null, "exited");
    expect(await listed()).toBe(true);
    // A computer stopped before the reason was recorded counts as not stopped by the person.
    await db.query("UPDATE computers SET stop_reason = NULL WHERE dot_id = $1", [dot.id]);
    expect(await listed()).toBe(true);
    await db.computers.setState(dot.id, "STOPPED", null, "user");
    await db.computers.setState(dot.id, "RUNNING", null);
    await db.computers.setState(dot.id, "STOPPED", null, "idle");
    expect(await listed()).toBe(true);
  });

  it("computers: stoppedWithAutomationBy lists the stopped computers whose automation is due by then, earliest first", async () => {
    const at = (iso: string) => Date.parse(iso);
    const seed = async (name: string, state: "STOPPED" | "RUNNING", nextAt: number | null, status: "IDLE" | "ERROR" | "DISABLED" | "CREATING" = "IDLE") => {
      const dot = await seedDot(db, name);
      await db.dots.setStatus(dot.id, status);
      await db.computers.setState(dot.id, state);
      if (nextAt !== null) await db.computers.setNextAutomation(dot.id, nextAt);
      return dot.id;
    };
    const soon = await seed("due-soon", "STOPPED", at("2030-01-01T10:00:00Z"));
    const missed = await seed("due-missed", "STOPPED", at("2029-12-31T00:00:00Z"));
    const later = await seed("due-later", "STOPPED", at("2030-01-01T11:00:00Z"));
    const none = await seed("due-none", "STOPPED", null);
    const running = await seed("due-running", "RUNNING", at("2030-01-01T10:00:00Z"));
    const broken = await seed("due-error", "STOPPED", at("2030-01-01T10:00:00Z"), "ERROR");
    const disabled = await seed("due-disabled", "STOPPED", at("2030-01-01T10:00:00Z"), "DISABLED");
    const creating = await seed("due-creating", "STOPPED", at("2030-01-01T10:00:00Z"), "CREATING");

    const ids = [soon, missed, later, none, running, broken, disabled, creating];
    const listed = (await db.computers.stoppedWithAutomationBy(new Date("2030-01-01T10:30:00Z"))).filter((id) => ids.includes(id));
    // The run that was missed comes first; the one due after the limit, the one with nothing due, the running
    // computer and the Dots the person has to look at are left alone.
    expect(listed).toEqual([missed, soon]);
    expect((await db.computers.stoppedWithAutomationBy(new Date("2030-01-01T11:00:00Z"))).filter((id) => ids.includes(id))).toEqual([missed, soon, later]);
  });

  it("events: guest events are idempotent on (dot_id, guest_seq) and queries filter", async () => {
    const dot = await seedDot(db, "charlie");
    const first = await db.events.insertGuest(dot.id, outbound(1, "agent.state", { state: "THINKING" }));
    expect(first?.guest_seq).toBe(1);
    expect(typeof first?.id).toBe("number");
    expect(first?.data.guest_event_id).toBe("evt-1");
    expect(await db.events.insertGuest(dot.id, outbound(1, "agent.state", { state: "THINKING" }))).toBeNull();
    const host = await db.events.insertHost(dot.id, "computer.state", { state: "RUNNING" });
    expect(host.source).toBe("host");
    expect(host.guest_seq).toBeNull();
    const all = await db.events.list({ dotId: dot.id });
    expect(all.map((e) => e.type)).toEqual(["agent.state", "computer.state"]);
    expect(await db.events.list({ dotId: dot.id, after: first!.id })).toHaveLength(1);
    expect(await db.events.list({ dotId: dot.id, types: ["computer.state"] })).toHaveLength(1);
    expect((await db.events.tail(dot.id, 1))[0]?.id).toBe(host.id);
    expect(await db.events.latestId()).toBeGreaterThanOrEqual(host.id);
  });

  it("events: one task's events come by data.task_id, with the other filters, and the index serves the query", async () => {
    const dot = await seedDot(db, "taskwise");
    const other = await seedDot(db, "taskless");
    let seq = 0;
    const guest = (id: string, type: OutboundEvent["type"], data: Record<string, unknown>) =>
      db.events.insertGuest(id, outbound(++seq, type, data));
    const started = await guest(dot.id, "task.started", { task_id: "task_a" });
    await guest(dot.id, "task.started", { task_id: "task_b" });
    await guest(dot.id, "tool.called", { task_id: "task_a", tool: "exec", permission: "exec.run", decision: "allow", ok: true, duration_ms: 3 });
    await guest(dot.id, "tool.called", { tool: "exec", permission: "exec.run", decision: "allow", ok: true, duration_ms: 3 });
    await guest(dot.id, "task.completed", { task_id: "task_a", summary: "done" });
    await guest(other.id, "task.started", { task_id: "task_a" });
    await db.events.insertHost(dot.id, "task.cancelled", { task_id: "task_b" });

    const types = (events: { type: string }[]) => events.map((e) => e.type);
    expect(types(await db.events.list({ dotId: dot.id, taskId: "task_a" }))).toEqual(["task.started", "tool.called", "task.completed"]);
    expect(types(await db.events.list({ dotId: dot.id, taskId: "task_b" }))).toEqual(["task.started", "task.cancelled"]);
    // The task id is matched whole, in this Dot, and an event without one belongs to no task.
    expect(await db.events.list({ dotId: dot.id, taskId: "task" })).toEqual([]);
    expect(await db.events.list({ dotId: dot.id, taskId: "" })).toEqual([]);
    expect((await db.events.list({ dotId: other.id, taskId: "task_a" })).map((e) => e.dot_id)).toEqual([other.id]);
    // With the other filters, and the limit counts the filtered events.
    expect(types(await db.events.list({ dotId: dot.id, taskId: "task_a", types: ["tool.called", "task.completed"] }))).toEqual(["tool.called", "task.completed"]);
    expect(types(await db.events.list({ dotId: dot.id, taskId: "task_a", after: started!.id }))).toEqual(["tool.called", "task.completed"]);
    expect(types(await db.events.list({ dotId: dot.id, taskId: "task_a", limit: 2 }))).toEqual(["task.started", "tool.called"]);

    // Among many events of other kinds the planner takes events_task_idx for the statement list() really issues: it is
    // captured from the repository, with every filter that can go with the task.
    await db.query(
      `INSERT INTO events (dot_id, type, data, source) SELECT $1, 'agent.state', '{"state":"IDLE"}'::jsonb, 'host' FROM generate_series(1, 6000)`,
      [dot.id],
    );
    await db.query("ANALYZE events");
    const issued: { sql: string; params: readonly unknown[] }[] = [];
    const recording = new EventsRepository({
      async query<R>(sql: string, params: readonly unknown[] = []) {
        issued.push({ sql, params });
        return db.query<R>(sql, params);
      },
    });
    for (const query of [
      { dotId: dot.id, taskId: "task_a" },
      { dotId: dot.id, taskId: "task_a", after: 0, types: ["tool.called", "task.completed"], limit: 100 },
    ]) {
      issued.length = 0;
      await recording.list(query);
      expect(issued).toHaveLength(1);
      const plan = await db.query<Record<string, string>>(`EXPLAIN ${issued[0]!.sql}`, issued[0]!.params);
      expect(plan.rows.map((r) => Object.values(r)[0]).join("\n"), JSON.stringify(query)).toContain("events_task_idx");
    }
  });

  it("events: newest first with a limit, and a tool filter that touches tool.called only", async () => {
    const dot = await seedDot(db, "newest-first");
    let seq = 0;
    const guest = (type: OutboundEvent["type"], data: Record<string, unknown>) => db.events.insertGuest(dot.id, outbound(++seq, type, data));
    const call = { permission: "exec.run", decision: "allow", ok: true, duration_ms: 1 };
    await guest("tool.called", { tool: "exec", ...call });
    const nav = await guest("tool.called", { tool: "browser_navigate", ...call });
    await guest("tool.called", { tool: "exec", ...call });
    const click = await guest("tool.called", { tool: "browser_click", ...call });
    const state = await guest("agent.state", { state: "IDLE" });
    await guest("tool.called", { tool: "exec", ...call });
    const ids = async (query: Parameters<typeof db.events.list>[0]) => (await db.events.list({ dotId: dot.id, ...query })).map((e) => e.id);
    const all = await ids({});
    expect(await ids({ order: "desc" })).toEqual([...all].reverse());
    expect(await ids({ order: "desc", limit: 2 })).toEqual([all[5]!, all[4]!]);
    // `tools` narrows tool.called to those tools and leaves other types as they are.
    expect(await ids({ types: ["tool.called"], tools: ["browser_navigate", "browser_click"] })).toEqual([nav!.id, click!.id]);
    expect(await ids({ types: ["tool.called", "agent.state"], tools: ["browser_click"], order: "desc" })).toEqual([state!.id, click!.id]);
    // The limit counts what is kept: the newest browser call is found past three calls of other tools.
    expect(await ids({ types: ["tool.called"], tools: ["browser_navigate", "browser_click"], order: "desc", limit: 1 })).toEqual([click!.id]);
    // With `after`, newest first still reads the events after it.
    expect(await ids({ after: nav!.id, order: "desc", limit: 1 })).toEqual([all[5]!]);
    // `before` goes on, older, from the oldest row of the last page, and it combines with the filters; it pages newest first only.
    expect(await ids({ order: "desc", limit: 2, before: all[4]! })).toEqual([all[3]!, all[2]!]);
    expect(await ids({ order: "desc", before: all[1]! })).toEqual([all[0]!]);
    expect(await ids({ order: "desc", before: all[0]! })).toEqual([]);
    expect(await ids({ types: ["tool.called"], tools: ["browser_navigate", "browser_click"], order: "desc", before: click!.id })).toEqual([nav!.id]);
    await expect(ids({ before: all[4]! })).rejects.toThrow(/order/);
  });

  it("events: the spend of a Dot sums the events that end a unit of spend, from a moment on", async () => {
    const dot = await seedDot(db, "spender");
    const other = await seedDot(db, "bystander");
    expect(await db.events.spentUsd(dot.id)).toBe(0);
    let seq = 0;
    const guest = (id: string, type: OutboundEvent["type"], data: Record<string, unknown>) =>
      db.events.insertGuest(id, outbound(++seq, type, data));
    await guest(dot.id, "task.progress", { task_id: "task_1", text: "a", spent_usd: 0.1 });
    await guest(dot.id, "task.completed", { task_id: "task_1", summary: "b", spent_usd: 0.2 });
    await guest(dot.id, "task.failed", { task_id: "task_2", error: "c", spent_usd: 0.5 });
    await guest(dot.id, "message.assistant", { text: "d", spent_usd: 0.25 });
    await guest(dot.id, "memory.updated", { conversations: 3, changed: true, spent_usd: 0.0625 });
    // An event that predates the report has none, and a type that reports no spend adds none.
    await guest(dot.id, "message.assistant", { text: "old" });
    await guest(dot.id, "automation.next_run", { next_run_at_ms: null, spent_usd: 99 });
    // Another Dot's, and a host event of the same type, are not this Dot's.
    await guest(other.id, "task.completed", { task_id: "task_9", summary: "z", spent_usd: 7 });
    await db.events.insertHost(dot.id, "task.completed", { task_id: "task_h", spent_usd: 50 });
    // 0.2 + 0.5 + 0.25 + 0.0625: the float sum is rounded to the hundred-millionth of a USD, the engine's resolution.
    expect(await db.events.spentUsd(dot.id)).toBe(1.0125);
    expect(await db.events.spentUsd(other.id)).toBe(7);

    const later = new Date(Date.now() + 60_000);
    expect(await db.events.spentUsd(dot.id, later)).toBe(0);
    expect(await db.events.spentUsd(dot.id, new Date(Date.now() - 60_000))).toBe(1.0125);
    await db.query("UPDATE events SET created_at = now() - interval '2 days' WHERE dot_id = $1 AND type = 'task.completed' AND source = 'guest'", [dot.id]);
    expect(await db.events.spentUsd(dot.id, new Date(Date.now() - 3_600_000))).toBe(0.8125);
    expect(await db.events.spentUsd(dot.id)).toBe(1.0125);
  });

  it("events: a float sum shows no noise", async () => {
    const dot = await seedDot(db, "noisy");
    await db.events.insertGuest(dot.id, outbound(1, "task.completed", { task_id: "t", summary: "s", spent_usd: 0.1 }));
    await db.events.insertGuest(dot.id, outbound(2, "message.assistant", { text: "m", spent_usd: 0.2 }));
    expect(await db.events.spentUsd(dot.id)).toBe(0.3);
  });

  it("tasks: the spend a guest reports only grows, counts after the task ended and stays inside its Dot", async () => {
    const dot = await seedDot(db, "meter");
    const stranger = await seedDot(db, "stranger");
    const task = await db.tasks.insert({ id: newId("task"), dotId: dot.id, description: "spend" });
    expect(task.spent_usd).toBe(0);
    await db.tasks.recordSpend(task.id, dot.id, 0.25);
    expect((await db.tasks.get(task.id))?.spent_usd).toBe(0.25);
    // A late or repeated event never lowers it.
    await db.tasks.recordSpend(task.id, dot.id, 0.1);
    expect((await db.tasks.get(task.id))?.spent_usd).toBe(0.25);
    await db.tasks.recordSpend(task.id, dot.id, 0.75);
    expect((await db.tasks.get(task.id))?.spent_usd).toBe(0.75);
    // Another Dot's guest cannot write it.
    await db.tasks.recordSpend(task.id, stranger.id, 50);
    expect((await db.tasks.get(task.id))?.spent_usd).toBe(0.75);
    // A cancelled task the guest keeps working on still spends.
    await db.tasks.transition(task.id, "CANCELLED");
    await db.tasks.recordSpend(task.id, dot.id, 1.5);
    expect((await db.tasks.get(task.id))?.spent_usd).toBe(1.5);
    expect((await db.tasks.listByDot(dot.id))[0]?.spent_usd).toBe(1.5);
  });

  it("tasks: a Dot's list is the newest TASK_LIST_LIMIT, and a limit given says otherwise", async () => {
    const dot = await seedDot(db, "task-list-limit");
    // Scheduled for tomorrow, so that no claim in the tests after this one takes them.
    const tomorrow = new Date(Date.now() + 86_400_000);
    const ids: string[] = [];
    for (let i = 0; i < TASK_LIST_LIMIT + 1; i++) {
      ids.push((await db.tasks.insert({ id: newId("task"), dotId: dot.id, description: `task ${i}`, scheduledAt: tomorrow })).id);
    }
    await createdInOrder(db, "tasks", ids);
    const listed = await db.tasks.listByDot(dot.id);
    expect(listed).toHaveLength(TASK_LIST_LIMIT);
    expect(listed.map((t) => t.id)).not.toContain(ids[0]);
    expect(await db.tasks.listByDot(dot.id, { limit: TASK_LIST_LIMIT + 1 })).toHaveLength(TASK_LIST_LIMIT + 1);
    // The oldest is reached by the cursor: a page goes on after the last task of the one before, and the end is empty.
    const rest = await db.tasks.listByDot(dot.id, { before: listed.at(-1)!.id });
    expect(rest.map((t) => t.id)).toEqual([ids[0]]);
    expect(await db.tasks.listByDot(dot.id, { before: ids[0]! })).toEqual([]);
    expect(await db.tasks.listByDot(dot.id, { before: "task_nobody" })).toEqual([]);
    const another = await seedDot(db, "task-list-cursor-elsewhere");
    expect(await db.tasks.listByDot(another.id, { before: listed.at(-1)!.id })).toEqual([]);
  });

  it("tasks: claim skips busy Dots, honours priority and scheduled_at, and never hands one Dot two tasks", async () => {
    const a = await seedDot(db, "delta");
    const b = await seedDot(db, "echo");
    const low = await db.tasks.insert({ id: newId("task"), dotId: a.id, description: "low", priority: 0 });
    const high = await db.tasks.insert({ id: newId("task"), dotId: a.id, description: "high", priority: 5 });
    await db.tasks.insert({
      id: newId("task"),
      dotId: b.id,
      description: "later",
      scheduledAt: new Date(Date.now() + 3_600_000),
    });

    // Two dispatchers race: only one task of Dot a may be claimed.
    const claims = await Promise.all([
      db.transaction((tx) => tx.tasks.claimNext()),
      db.transaction((tx) => tx.tasks.claimNext()),
    ]);
    const claimed = claims.filter((c) => c !== null);
    expect(claimed).toHaveLength(1);
    expect(claimed[0]?.task.id).toBe(high.id);
    expect(claimed[0]?.task.status).toBe("RUNNING");
    expect(claimed[0]?.run.delivered_at).toBeNull();
    // The claim stored the task.created for the guest in the same transaction.
    expect(claimed[0]?.inbound.event).toMatchObject({
      type: "task.created",
      data: { task_id: high.id, description: "high", priority: 5 },
    });
    expect((await db.inbound.pending(a.id)).map((r) => r.event.id)).toEqual([claimed[0]!.inbound.event.id]);

    // Dot a is busy and Dot b's task is not due.
    expect(await db.transaction((tx) => tx.tasks.claimNext())).toBeNull();
    expect(await db.tasks.hasWork(b.id, new Date())).toBe(false);
    expect(await db.tasks.hasWork(a.id, new Date())).toBe(true);
    expect((await db.tasks.activeForDot(a.id)).map((t) => t.id)).toEqual([high.id]);

    // Delivered: the run says so and nothing is pending any more.
    const sent = claimed[0]!.inbound.event.id;
    expect(await db.inbound.beginSend(sent)).toBe("send");
    await db.inbound.markDelivered(sent);
    expect((await db.tasks.runs(high.id))[0]?.delivered_at).not.toBeNull();
    expect(await db.inbound.pending(a.id)).toEqual([]);

    const done = await db.tasks.transition(high.id, "COMPLETED", { summary: "ok" });
    expect(done?.previous).toBe("RUNNING");
    expect(done?.task.status).toBe("COMPLETED");
    expect(done?.task.finished_at).not.toBeNull();
    expect((await db.tasks.runs(high.id)).at(-1)?.outcome).toBe("completed");
    // Terminal tasks do not move again, and a guest cannot move another Dot's task.
    expect(await db.tasks.transition(high.id, "FAILED", { error: "late" })).toBeNull();
    expect(await db.tasks.transition(low.id, "RUNNING", { dotId: b.id })).toBeNull();

    const next = await db.transaction((tx) => tx.tasks.claimNext());
    expect(next?.task.id).toBe(low.id);
    expect((await db.tasks.listByDot(a.id)).map((t) => t.id)).toContain(low.id);
    expect((await db.tasks.listByDot(a.id, { status: "COMPLETED" })).map((t) => t.id)).toEqual([high.id]);
  });

  it("inbound: a cancel and a send of the same task.created decide atomically who wins", async () => {
    const dot = await seedDot(db, "kilo");
    const event = (id: string) =>
      ({ id, type: "task.created", ts: new Date().toISOString(), data: { task_id: "x", description: "d", priority: 0 } }) as const;

    // The cancel wins: the send that comes after finds the row dropped and the task ended.
    const first = await db.tasks.insert({ id: newId("task"), dotId: dot.id, description: "first" });
    await db.tasks.transition(first.id, "RUNNING");
    await db.inbound.enqueue(dot.id, event("evt_first"), { taskId: first.id });
    await db.tasks.transition(first.id, "CANCELLED");
    expect(await db.inbound.dropUnsent(first.id, "cancelled")).toBe(true);
    expect(await db.inbound.beginSend("evt_first")).toBe("skip");

    // The send wins: the cancel finds a row the guest may hold, so it must tell the guest.
    const second = await db.tasks.insert({ id: newId("task"), dotId: dot.id, description: "second" });
    await db.tasks.transition(second.id, "RUNNING");
    await db.inbound.enqueue(dot.id, event("evt_second"), { taskId: second.id });
    expect(await db.inbound.beginSend("evt_second")).toBe("send");
    await db.tasks.transition(second.id, "CANCELLED");
    expect(await db.inbound.dropUnsent(second.id, "cancelled")).toBe(false);
    // Its retry is never sent again for a task that ended: it is dropped instead.
    expect(await db.inbound.beginSend("evt_second")).toBe("skip");
    expect((await db.inbound.get("evt_second"))?.dropped_at).not.toBeNull();
    expect(await db.inbound.pending(dot.id)).toEqual([]);
  });

  it("inbound: rows wait in order, failures count, and only due rows are retried", async () => {
    const dot = await seedDot(db, "lima");
    const message = (id: string) => ({ id, type: "user.message", ts: new Date().toISOString(), data: { text: id } }) as const;
    await db.inbound.enqueue(dot.id, message("evt_m1"));
    await db.inbound.enqueue(dot.id, message("evt_m2"));
    expect((await db.inbound.pending(dot.id)).map((r) => r.event.id)).toEqual(["evt_m1", "evt_m2"]);
    expect(await db.tasks.hasWork(dot.id, new Date())).toBe(true);
    expect(await db.inbound.dueDots(3)).toContain(dot.id);
    expect(await db.inbound.recordFailure("evt_m1", "refused", new Date(Date.now() + 3_600_000))).toBe(1);
    expect(await db.inbound.recordFailure("evt_m2", "refused", new Date(Date.now() + 3_600_000))).toBe(1);
    expect(await db.inbound.dueDots(3)).not.toContain(dot.id);
    await db.inbound.markDelivered("evt_m1");
    await db.inbound.markDelivered("evt_m2");
    expect(await db.tasks.hasWork(dot.id, new Date())).toBe(false);
  });

  it("ending a task expires its pending approvals; a request for an ended task is never pending", async () => {
    const dot = await seedDot(db, "mike");
    const task = await db.tasks.insert({ id: newId("task"), dotId: dot.id, description: "asks" });
    await db.tasks.transition(task.id, "RUNNING");
    const request = (approvalId: string) => ({
      approval_id: approvalId,
      task_id: task.id,
      tool: "files_write",
      permission: "files.write" as const,
      arguments: {},
      reason: "needs a decision",
    });
    await db.approvals.insertRequested(dot.id, request("apr_waiting"));
    const cancelled = await db.tasks.transition(task.id, "CANCELLED");
    expect(cancelled?.previous).toBe("RUNNING");
    expect((await db.approvals.get("apr_waiting"))?.status).toBe("expired");
    expect((await db.approvals.list({ status: "pending", dotId: dot.id })).map((a) => a.id)).toEqual([]);
    expect(await db.approvals.resolve("apr_waiting", "approved", null)).toBeNull();
    expect((await db.approvals.insertRequested(dot.id, request("apr_late"), "expired"))?.status).toBe("expired");
  });

  it("event ids become visible in id order: an insert waits for a transaction that drew a lower id", async () => {
    const dot = await seedDot(db, "november");
    let release!: () => void;
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    let lowId = 0;
    const slow = db.transaction(async (tx) => {
      lowId = (await tx.events.insertGuest(dot.id, outbound(1, "automation.next_run", { next_run_at_ms: null })))!.id;
      await held;
    });
    // Let the transaction draw its id before the autocommit insert starts.
    while (lowId === 0) await new Promise((resolve) => setTimeout(resolve, 5));
    let highDone = false;
    const fast = db.events.insertHost(dot.id, "computer.started", {}).then((event) => {
      highDone = true;
      return event;
    });
    await new Promise((resolve) => setTimeout(resolve, 100));
    // Committing 11 before 10 would let a stream resumed after 11 skip 10 for good.
    expect(highDone).toBe(false);
    release();
    await slow;
    expect((await fast).id).toBeGreaterThan(lowId);
  });

  it("one server per database: the instance lock is held for the life of the handle", async () => {
    expect(await db.holdInstanceLock()).toBe(true);
    if (kind === "pg") {
      const { Database } = await import("../src/index.js");
      const second = await Database.open({ target: { kind: "pg", url: t.url! }, masterKey: new Uint8Array(32) });
      try {
        expect(await second.holdInstanceLock()).toBe(false);
      } finally {
        await second.close();
      }
    }
  });

  it("approvals: replayed requests are ignored and a resolution happens once", async () => {
    const dot = await seedDot(db, "foxtrot");
    const data = {
      approval_id: newId("apr"),
      tool: "browser_identity_delete",
      permission: "browser.identity.delete" as const,
      arguments: { identity_id: "x-abc123" },
      reason: "cleanup",
    };
    expect((await db.approvals.insertRequested(dot.id, data))?.status).toBe("pending");
    expect(await db.approvals.insertRequested(dot.id, data)).toBeNull();
    expect((await db.approvals.get(data.approval_id))?.arguments).toEqual({ identity_id: "x-abc123" });
    expect((await db.approvals.list({ status: "pending", dotId: dot.id })).map((a) => a.id)).toEqual([data.approval_id]);
    expect((await db.approvals.resolve(data.approval_id, "approved", "fine"))?.note).toBe("fine");
    expect(await db.approvals.resolve(data.approval_id, "rejected", null)).toBeNull();
  });

  it("approvals: a list is the oldest APPROVAL_LIST_LIMIT in the order they were asked, and a limit given says otherwise", async () => {
    const dot = await seedDot(db, "approval-list-limit");
    const ids: string[] = [];
    for (let i = 0; i < APPROVAL_LIST_LIMIT + 1; i++) {
      const approval_id = newId("apr");
      ids.push(approval_id);
      await db.approvals.insertRequested(dot.id, { approval_id, tool: "exec", permission: "computer.exec" as const, arguments: {}, reason: `ask ${i}` });
    }
    await createdInOrder(db, "approvals", ids);
    const listed = await db.approvals.list({ dotId: dot.id });
    expect(listed).toHaveLength(APPROVAL_LIST_LIMIT);
    expect(listed.map((a) => a.id)).not.toContain(ids[APPROVAL_LIST_LIMIT]);
    expect(await db.approvals.list({ dotId: dot.id, limit: APPROVAL_LIST_LIMIT + 1 })).toHaveLength(APPROVAL_LIST_LIMIT + 1);
  });

  it("approvals: newest first by the time of their last change, several statuses at once, and a cursor pages on without a gap", async () => {
    const dot = await seedDot(db, "approval-newest");
    const ask = (reason: string) => {
      const approval_id = newId("apr");
      return db.approvals.insertRequested(dot.id, { approval_id, tool: "exec", permission: "computer.exec" as const, arguments: {}, reason }).then(() => approval_id);
    };
    const first = await ask("first");
    const second = await ask("second");
    const third = await ask("third");
    const fourth = await ask("fourth");
    // Times set by hand: the clock of two statements in a row may be the same.
    await db.approvals.resolve(third, "approved", null);
    await db.approvals.resolve(first, "rejected", null);
    await db.query("UPDATE approvals SET status = 'expired' WHERE id = $1", [second]);
    const minutes = (m: number) => new Date(Date.UTC(2026, 0, 1, 12, m));
    const at: [string, number, number | null][] = [[first, 0, 30], [second, 1, 40], [third, 2, 20], [fourth, 3, null]];
    for (const [id, created, resolved] of at) {
      await db.query("UPDATE approvals SET created_at = $2, resolved_at = $3 WHERE id = $1", [id, minutes(created), resolved === null ? null : minutes(resolved)]);
    }
    // Answered last first (second at :40, first at :30, third at :20); the pending fourth sorts by when it was asked.
    const ids = async (options: Parameters<typeof db.approvals.list>[0]) => (await db.approvals.list({ dotId: dot.id, ...options })).map((a) => a.id);
    expect(await ids({ order: "asc" })).toEqual([first, second, third, fourth]);
    expect(await ids({ order: "desc", status: ["approved", "rejected", "expired"] })).toEqual([second, first, third]);
    expect(await ids({ order: "desc", status: "pending" })).toEqual([fourth]);
    expect(await ids({ status: ["approved", "expired"] })).toEqual([second, third]);
    // The newest ones survive a limit, which is the point of the order: an old list does not hide the new answers.
    expect(await ids({ order: "desc", status: ["approved", "rejected", "expired"], limit: 2 })).toEqual([second, first]);
    // A cursor is the id of the last row seen; pages cover the list once, in order.
    const answered = ["approved", "rejected", "expired"] as const;
    expect(await ids({ order: "desc", status: answered, limit: 2, before: first })).toEqual([third]);
    expect(await ids({ order: "desc", status: answered, limit: 2, before: third })).toEqual([]);
    expect(await ids({ order: "desc", status: answered, before: "apr_nobody" })).toEqual([]);
    await expect(db.approvals.list({ dotId: dot.id, before: first })).rejects.toThrow(/order/);
  });

  it("approvals: the newest of more than APPROVAL_LIST_LIMIT are listed when asked for, and no pending one uses a place of the answered", async () => {
    const dot = await seedDot(db, "approval-newest-limit");
    const ids: string[] = [];
    for (let i = 0; i < APPROVAL_LIST_LIMIT + 3; i++) {
      const approval_id = newId("apr");
      ids.push(approval_id);
      await db.approvals.insertRequested(dot.id, { approval_id, tool: "exec", permission: "computer.exec" as const, arguments: {}, reason: `ask ${i}` });
    }
    // The last three are answered; the others stay pending.
    for (const id of ids.slice(-3)) await db.approvals.resolve(id, "approved", null);
    const answered = await db.approvals.list({ dotId: dot.id, status: ["approved", "rejected", "expired"], order: "desc" });
    expect(answered).toHaveLength(3);
    expect(answered.map((a) => a.id).sort()).toEqual(ids.slice(-3).sort());
    // The pending ones are listed by their status and do not take a place of the limit: all but the answered three, up to it.
    expect(await db.approvals.list({ dotId: dot.id, status: "pending" })).toHaveLength(APPROVAL_LIST_LIMIT);
  });

  it("approvals: found by the end of their id, in lowercase, within one Dot, resolved ones included", async () => {
    const dot = await seedDot(db, "foxtrot-suffix");
    const other = await seedDot(db, "foxtrot-other");
    const data = (id: string) => ({ approval_id: id, tool: "exec", permission: "browser.identity.delete" as const, arguments: {}, reason: "test" });
    await db.approvals.insertRequested(dot.id, data("apr_aaaaaaaaaaaaaaaaaaaaaaaaaa9bntc"));
    await db.approvals.insertRequested(dot.id, data("apr_bbbbbbbbbbbbbbbbbbbbbbbbbb9bntc"));
    await db.approvals.insertRequested(dot.id, data("apr_cccccccccccccccccccccccccccccc"));
    await db.approvals.insertRequested(other.id, data("apr_dddddddddddddddddddddddddd9bntc"));
    await db.approvals.resolve("apr_aaaaaaaaaaaaaaaaaaaaaaaaaa9bntc", "approved", null);
    const found = await db.approvals.endingWith(dot.id, "9BNTC");
    expect(found.map((a) => [a.id.slice(0, 5), a.status])).toEqual([
      ["apr_a", "approved"],
      ["apr_b", "pending"],
    ]);
    expect(await db.approvals.endingWith(dot.id, "zzzzz")).toEqual([]);
    expect((await db.approvals.endingWith(other.id, "9bntc")).map((a) => a.id)).toEqual(["apr_dddddddddddddddddddddddddd9bntc"]);
  });

  it("secrets: encrypted at rest, per-Dot key wins over the global one", async () => {
    const dot = await seedDot(db, "golf");
    expect(await db.secrets.openRouterKey(dot.id)).toBeNull();
    await db.secrets.put("global", "openrouter_api_key", "sk-global");
    expect(await db.secrets.openRouterKey(dot.id)).toBe("sk-global");
    await db.secrets.put(dot.id, "openrouter_api_key", "sk-dot");
    await db.secrets.put(dot.id, "openrouter_api_key", "sk-dot-2");
    expect(await db.secrets.openRouterKey(dot.id)).toBe("sk-dot-2");
    const { rows } = await db.query<{ value_enc: Uint8Array }>("SELECT value_enc FROM secrets");
    for (const row of rows) expect(Buffer.from(row.value_enc).toString("latin1")).not.toContain("sk-");
    expect(await db.secrets.delete(dot.id, "openrouter_api_key")).toBe(true);
    expect(await db.secrets.openRouterKey(dot.id)).toBe("sk-global");
    expect(await db.holdsEncryptedValues()).toBe(true);
  });

  it("deleting a Dot cascades to its computer, tasks and approvals but keeps its events", async () => {
    const dot = await seedDot(db, "hotel");
    await db.tasks.insert({ id: newId("task"), dotId: dot.id, description: "x" });
    await db.events.insertHost(dot.id, "dot.deleted", { name: "hotel" });
    expect(await db.dots.delete(dot.id)).toBe(true);
    expect(await db.computers.get(dot.id)).toBeNull();
    expect(await db.tasks.listByDot(dot.id)).toEqual([]);
    expect(await db.events.list({ dotId: dot.id })).toHaveLength(1);
  });

  it("deleting a Dot deletes its own secrets and leaves the global and other Dots' secrets", async () => {
    const gone = await seedDot(db, "hotel-secrets");
    const kept = await seedDot(db, "india-secrets");
    await db.secrets.put("global", "openrouter_api_key", "sk-global");
    await db.secrets.put(gone.id, "openrouter_api_key", "sk-gone");
    await db.secrets.put(gone.id, "telegram_bot_token", "123:abc");
    await db.secrets.put(kept.id, "openrouter_api_key", "sk-kept");
    expect(await db.dots.delete(gone.id)).toBe(true);
    const { rows } = await db.query<{ scope: string; name: string }>("SELECT scope, name FROM secrets ORDER BY scope, name");
    expect(rows.filter((r) => r.scope === gone.id)).toEqual([]);
    expect(await db.secrets.openRouterKey(gone.id)).toBe("sk-global");
    expect(await db.secrets.openRouterKey(kept.id)).toBe("sk-kept");
  });

  it("a transaction rolls back when its function throws, and refuses the outer repositories inside it", async () => {
    await expect(
      db.transaction(async (tx) => {
        await tx.dots.insert({ id: newId("dot"), config: parseDotConfig(yaml("india")), status: "CREATING" });
        throw new Error("abort");
      }),
    ).rejects.toThrow("abort");
    expect(await db.dots.resolve("india")).toBeNull();
    await expect(db.transaction(() => db.dots.list())).rejects.toBeInstanceOf(TransactionMisuseError);
  });
});

describe.each(testAdapters())("an empty database on %s", { timeout: SETUP_TIMEOUT }, (kind) => {
  it("holds no encrypted values", async () => {
    const t = await createTestDatabase(kind);
    try {
      expect(await t.db.holdsEncryptedValues()).toBe(false);
    } finally {
      await t.drop();
    }
  });
});
