/**
 * Event shapes of architecture section 5.4: inbound (host to guest), outbound
 * (guest to host, persisted in the guest outbox) and the control plane's own
 * host events.
 */
import { z } from "zod";
import { AGENT_STATES, type AgentState, type EventSource, type VmState } from "./states.js";
import { MAX_RUN_AT_MS } from "./protocol.js";
import { PERMISSIONS, type Permission } from "./tools.js";

/**
 * The most events one read of the event log returns (`GET /api/dots/:id/events?limit=`): the store clamps to it, the
 * route refuses more, and a client that pages through the log asks for exactly this many, so a page shorter than it
 * is the last one.
 */
export const MAX_EVENT_PAGE = 1000;

export const INBOUND_EVENT_TYPES = ["user.message", "task.created", "approval.received", "system.event"] as const;
export type InboundEventType = (typeof INBOUND_EVENT_TYPES)[number];

/**
 * The `system.event` name the host sends when the user cancels a task the
 * guest already has; `data` is `{ task_id }`. The contract has no inbound
 * cancel type, so it travels as a system event.
 */
export const TASK_CANCELLED_SYSTEM_EVENT = "task.cancelled";

/** The events after which a Dot's list of browser identities, or one of them, has changed. */
export const IDENTITY_EVENT_TYPES = [
  "browser.identity.created",
  "browser.identity.deleted",
  "browser.identity.launched",
  "browser.identity.closed",
] as const;

export const OUTBOUND_EVENT_TYPES = [
  "agent.started",
  "agent.state",
  "message.assistant",
  "task.started",
  "task.progress",
  "task.completed",
  "task.failed",
  "approval.requested",
  "tool.called",
  ...IDENTITY_EVENT_TYPES,
  "automation.next_run",
  "memory.updated",
] as const;
export type OutboundEventType = (typeof OUTBOUND_EVENT_TYPES)[number];

export const HOST_EVENT_TYPES = [
  "dot.created",
  "dot.updated",
  "dot.deleted",
  "computer.state",
  "computer.started",
  "computer.stopped",
  "task.created",
  "task.cancelled",
  "approval.resolved",
  "guest.event.refused",
  "channel.status",
  "channel.peer.paired",
  "channel.changed",
] as const;
export type HostEventType = (typeof HOST_EVENT_TYPES)[number];

/** Every type that can appear in the host event log. */
export type EventType = OutboundEventType | HostEventType;

/**
 * Every type name a row of the host event log can carry: the guest's, the host's, and the person's own
 * `user.message` (stored by the host, so it is neither). The filter of `GET /api/dots/:id/events?types=` takes these.
 */
export const STORED_EVENT_TYPES: readonly string[] = [...OUTBOUND_EVENT_TYPES, ...HOST_EVENT_TYPES, "user.message"];

export function isStoredEventType(value: unknown): value is string {
  return typeof value === "string" && STORED_EVENT_TYPES.includes(value);
}

/** The messaging channels the control plane can bridge a Dot to. */
export const CHANNEL_KINDS = ["telegram", "whatsapp"] as const;
export type ChannelKind = (typeof CHANNEL_KINDS)[number];

/** Where a channel's connection stands; `needs_relink` is a login the person has to redo. */
export const CHANNEL_STATUSES = ["connecting", "connected", "needs_relink", "error"] as const;
export type ChannelStatus = (typeof CHANNEL_STATUSES)[number];

/** What the person did to a channel that its status does not say: paused it, resumed it, or removed it with its credentials and people. */
export const CHANNEL_CHANGES = ["paused", "resumed", "removed"] as const;
export type ChannelChange = (typeof CHANNEL_CHANGES)[number];

/**
 * Where a user message came from when it did not come from the web, the CLI or the SDK: the channel
 * binding of a Dot, the chat on that channel and the message's id there. It is stored in the data of
 * the `user.message` event and nowhere else, so the event log owns the fact and a reply is routed back
 * by it. The guest never sees it. Absent means the control plane's own API.
 */
export interface MessageOrigin {
  channel: ChannelKind;
  binding_id: string;
  chat_id: string;
  external_id: string;
}

const originField = z.string().min(1).max(256);

export const messageOriginSchema = z.strictObject({
  channel: z.enum(CHANNEL_KINDS),
  binding_id: originField,
  chat_id: originField,
  external_id: originField,
});

/** The origin in `value` (the data of a stored `user.message`), or null when there is none or it is not one. */
export function parseMessageOrigin(value: unknown): MessageOrigin | null {
  const result = messageOriginSchema.safeParse(value);
  return result.success ? result.data : null;
}

export type ApprovalDecision = "approve" | "reject";
export type PolicyDecision = "allow" | "ask" | "deny";

export interface InboundEventDataMap {
  "user.message": { text: string };
  "task.created": { task_id: string; description: string; priority: number };
  "approval.received": { approval_id: string; decision: ApprovalDecision; note?: string };
  "system.event": { name: string; data: Record<string, unknown> };
}

export interface BrowserIdentityEventData {
  identity_id: string;
  name: string;
}

export interface ApprovalRequestedData {
  approval_id: string;
  task_id?: string;
  tool: string;
  permission: Permission;
  arguments: Record<string, unknown>;
  reason: string;
}

/**
 * Model spend in USD, on the events that report it (architecture section 5.4): what the
 * session of the event has spent so far. A task's events carry the task's spend (it only
 * grows), the chat's `message.assistant` the spend of the turn that answered, a `memory.updated` the spend of the
 * memory passes since the last one. The engine always sends it; it is optional because events logged before it existed have none.
 */
export interface SpentUsd {
  spent_usd?: number;
}

/**
 * The events a usage total sums `spent_usd` over: each one ends a unit of spend and carries the whole
 * of it. `task.progress` also carries `spent_usd`, but it is a running value of a task that ends with
 * one of these, so summing it would count the same money twice.
 */
export const USAGE_EVENT_TYPES = ["task.completed", "task.failed", "message.assistant", "memory.updated"] as const satisfies readonly OutboundEventType[];

/** The longest `target` of a `tool.called` event, in characters: code points, which is how zod 4 measures a string (the engine's copy is in nanobot/dots/protocol.py). */
export const TOOL_TARGET_MAX = 160;

export interface OutboundEventDataMap {
  /**
   * The agent process started (a boot, or a restart by systemd inside a
   * running VM). It keeps the OpenRouter key in memory only, so the host
   * pushes the key again when it sees this.
   */
  "agent.started": Record<string, never>;
  "agent.state": { state: AgentState };
  "message.assistant": { text: string; in_reply_to?: string } & SpentUsd;
  "task.started": { task_id: string };
  "task.progress": { task_id: string; text: string } & SpentUsd;
  "task.completed": { task_id: string; summary: string } & SpentUsd;
  "task.failed": { task_id: string; error: string } & SpentUsd;
  "approval.requested": ApprovalRequestedData;
  "tool.called": {
    task_id?: string;
    tool: string;
    /** Empty when the model called a tool the registry does not know (always denied). */
    permission: string;
    decision: PolicyDecision;
    ok: boolean;
    duration_ms: number;
    /**
     * What the call acted on, in one line of at most TOOL_TARGET_MAX characters: the first
     * line of a command, a path, a search term, an action and a name (architecture section 8.3).
     * Never what a person typed into a program or the text a browser field was given. Absent for a
     * call that never started (denied, not offered) and for a tool with nothing to name.
     */
    target?: string;
    /**
     * The call started a terminal session (`exec` with `tty`): a client shows "Started a terminal
     * session: python3" and not "ran a command". Absent for every other call.
     */
    tty?: true;
    /**
     * The agent stopped while the call ran, so its outcome is unknown and it
     * was not run again (architecture section 8.7). `ok` is false and
     * `duration_ms` is 0.
     */
    interrupted?: true;
  };
  "browser.identity.created": BrowserIdentityEventData;
  "browser.identity.deleted": BrowserIdentityEventData;
  "browser.identity.launched": BrowserIdentityEventData;
  "browser.identity.closed": BrowserIdentityEventData;
  /**
   * When the earliest enabled automation of the Dot is next due, in milliseconds since the epoch, or null when none is
   * (no job, all paused, or only one-time jobs that ran). The engine sends it each time that changes, so the last one
   * is what is true; the host keeps it (`computers.next_automation_at`) to wake a stopped computer shortly before the
   * run and to not put one to sleep that is about to need it (architecture section 9.5). A value in the past is a
   * run the engine has not made yet: it makes it when it starts.
   */
  "automation.next_run": { next_run_at_ms: number | null };
  /**
   * The Dot brought its MEMORY.md (what it is given about the person in every prompt) up to date from what was said
   * since the last time, once it had been quiet for a while (the engine's `memory_update.py`): how many conversation
   * files it took messages from, whether MEMORY.md changed, and what the pass cost (with the cost of any pass that
   * failed since the last of these).
   */
  "memory.updated": { conversations: number; changed: boolean } & SpentUsd;
}

/**
 * Data of host events. The contract fixes only `computer.state`; the others
 * carry whatever identifies their subject, and the dot id is always the
 * event's own `dot_id` column.
 */
/**
 * A message of the guest's event stream that the host refused: its schema did not take it, or it was not JSON. The
 * stream goes on past it and the event is not stored, so this record (the host event `guest.event.refused`) is the
 * only trace of it. It names what can be named: the `seq` and `type` the message carried when it carried them (null
 * otherwise), and the problem in words, at most REFUSED_PROBLEM_MAX characters.
 */
export interface RefusedEvent {
  seq: number | null;
  type: string | null;
  problem: string;
}

/** The longest `problem` of a refused event, in characters. */
export const REFUSED_PROBLEM_MAX = 300;

export interface HostEventDataMap {
  "dot.created": { name: string; [key: string]: unknown };
  "dot.updated": { name: string; [key: string]: unknown };
  "dot.deleted": { name: string; [key: string]: unknown };
  "computer.state": { state: VmState };
  "computer.started": Record<string, unknown>;
  "computer.stopped": Record<string, unknown>;
  "task.created": { task_id: string; description: string; priority: number };
  "task.cancelled": { task_id: string };
  /**
   * `always` marks an approval answered with "always allow": the Dot's config now allows the permission (`dot.updated`
   * follows). `task_id` is the task the approval was asked in, when it was asked in one: the events of a task
   * (`GET .../events?task_id=`) then hold the answer next to the request.
   */
  "approval.resolved": { approval_id: string; decision: ApprovalDecision; task_id?: string; note?: string; always?: true };
  /** A channel's connection changed (`detail` is a reason for `error`, never a credential). */
  "channel.status": { kind: ChannelKind; status: ChannelStatus; detail?: string };
  /** A person was paired to the Dot's channel; `peer_id` is the channel's own id for them. */
  "channel.peer.paired": { kind: ChannelKind; peer_id: string; label: string };
  /** The person paused, resumed or removed a channel (a removed one has no status left to report): every view of the channel follows it. */
  "channel.changed": { kind: ChannelKind; change: ChannelChange };
  "guest.event.refused": RefusedEvent;
}

export type InboundEvent<T extends InboundEventType = InboundEventType> = {
  [K in T]: { id: string; type: K; ts: string; data: InboundEventDataMap[K] };
}[T];

export type OutboundEvent<T extends OutboundEventType = OutboundEventType> = {
  [K in T]: { seq: number; id: string; type: K; ts: string; data: OutboundEventDataMap[K] };
}[T];

export type HostEvent<T extends HostEventType = HostEventType> = {
  [K in T]: { type: K; data: HostEventDataMap[K] };
}[T];

/** A row of the host `events` table, as the API returns it and the SSE stream sends it. */
export interface StoredEvent {
  id: number;
  dot_id: string;
  type: EventType;
  data: Record<string, unknown>;
  source: EventSource;
  guest_seq: number | null;
  created_at: string;
}

const isoTimestamp = z.iso.datetime({ offset: true });
const nonEmpty = z.string().min(1);

const inboundBase = { id: nonEmpty, ts: isoTimestamp };

export const inboundEventSchema = z.discriminatedUnion("type", [
  z.object({ ...inboundBase, type: z.literal("user.message"), data: z.object({ text: nonEmpty }) }),
  z.object({
    ...inboundBase,
    type: z.literal("task.created"),
    data: z.object({ task_id: nonEmpty, description: nonEmpty, priority: z.number().int() }),
  }),
  z.object({
    ...inboundBase,
    type: z.literal("approval.received"),
    data: z.object({
      approval_id: nonEmpty,
      decision: z.enum(["approve", "reject"]),
      note: z.string().optional(),
    }),
  }),
  z.object({
    ...inboundBase,
    type: z.literal("system.event"),
    data: z.object({ name: nonEmpty, data: z.record(z.string(), z.unknown()) }),
  }),
]);

/** Validate an inbound event; throws with every problem listed. */
export function parseInboundEvent(value: unknown): InboundEvent {
  const result = inboundEventSchema.safeParse(value);
  if (!result.success) {
    const detail = result.error.issues
      .map((issue) => `${issue.path.map(String).join(".") || "<root>"}: ${issue.message}`)
      .join("; ");
    throw new Error(`invalid inbound event: ${detail}`);
  }
  return result.data as InboundEvent;
}

const outboundBase = { seq: z.number().int().positive(), id: nonEmpty, ts: isoTimestamp };
const identityData = z.object({ identity_id: nonEmpty, name: z.string() });
const permission = z.enum(PERMISSIONS);
const spentUsd = z.number().nonnegative().optional();

export const outboundEventSchema = z.discriminatedUnion("type", [
  z.object({ ...outboundBase, type: z.literal("agent.started"), data: z.object({}) }),
  z.object({ ...outboundBase, type: z.literal("agent.state"), data: z.object({ state: z.enum(AGENT_STATES) }) }),
  z.object({
    ...outboundBase,
    type: z.literal("message.assistant"),
    data: z.object({ text: z.string(), in_reply_to: z.string().optional(), spent_usd: spentUsd }),
  }),
  z.object({ ...outboundBase, type: z.literal("task.started"), data: z.object({ task_id: nonEmpty }) }),
  z.object({
    ...outboundBase,
    type: z.literal("task.progress"),
    data: z.object({ task_id: nonEmpty, text: z.string(), spent_usd: spentUsd }),
  }),
  z.object({
    ...outboundBase,
    type: z.literal("task.completed"),
    data: z.object({ task_id: nonEmpty, summary: z.string(), spent_usd: spentUsd }),
  }),
  z.object({
    ...outboundBase,
    type: z.literal("task.failed"),
    data: z.object({ task_id: nonEmpty, error: z.string(), spent_usd: spentUsd }),
  }),
  z.object({
    ...outboundBase,
    type: z.literal("approval.requested"),
    data: z.object({
      approval_id: nonEmpty,
      task_id: z.string().optional(),
      tool: nonEmpty,
      permission,
      arguments: z.record(z.string(), z.unknown()),
      reason: z.string(),
    }),
  }),
  z.object({
    ...outboundBase,
    type: z.literal("tool.called"),
    data: z.object({
      task_id: z.string().optional(),
      tool: nonEmpty,
      permission: z.string(),
      decision: z.enum(["allow", "ask", "deny"]),
      ok: z.boolean(),
      duration_ms: z.number().nonnegative(),
      target: z
        .string()
        .min(1)
        .max(TOOL_TARGET_MAX)
        .regex(/^[^\r\n]*$/, "a target is one line")
        .optional(),
      tty: z.literal(true).optional(),
      interrupted: z.literal(true).optional(),
    }),
  }),
  z.object({ ...outboundBase, type: z.literal("browser.identity.created"), data: identityData }),
  z.object({ ...outboundBase, type: z.literal("browser.identity.deleted"), data: identityData }),
  z.object({ ...outboundBase, type: z.literal("browser.identity.launched"), data: identityData }),
  z.object({ ...outboundBase, type: z.literal("browser.identity.closed"), data: identityData }),
  z.object({
    ...outboundBase,
    type: z.literal("automation.next_run"),
    data: z.object({ next_run_at_ms: z.number().int().nonnegative().max(MAX_RUN_AT_MS).nullable() }),
  }),
  z.object({
    ...outboundBase,
    type: z.literal("memory.updated"),
    data: z.object({ conversations: z.number().int().nonnegative(), changed: z.boolean(), spent_usd: spentUsd }),
  }),
]);

/** Validate an outbound event read from the guest stream; throws with every problem listed. */
export function parseOutboundEvent(value: unknown): OutboundEvent {
  const result = outboundEventSchema.safeParse(value);
  if (!result.success) {
    const detail = result.error.issues
      .map((issue) => `${issue.path.map(String).join(".") || "<root>"}: ${issue.message}`)
      .join("; ");
    throw new Error(`invalid outbound event: ${detail}`);
  }
  return result.data as OutboundEvent;
}

export function isInboundEventType(value: unknown): value is InboundEventType {
  return typeof value === "string" && (INBOUND_EVENT_TYPES as readonly string[]).includes(value);
}

export function isOutboundEventType(value: unknown): value is OutboundEventType {
  return typeof value === "string" && (OUTBOUND_EVENT_TYPES as readonly string[]).includes(value);
}

export function isHostEventType(value: unknown): value is HostEventType {
  return typeof value === "string" && (HOST_EVENT_TYPES as readonly string[]).includes(value);
}
