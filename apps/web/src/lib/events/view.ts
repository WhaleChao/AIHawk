/**
 * How one row of a Dot's event log reads on the Activity page: a family to filter by, a tone, a title and one line of
 * detail. Every type the log can hold has its own description, written against the data the contract
 * (`packages/shared/src/events.ts`) says that type carries, so a renamed field or a new type is a compile error here,
 * and a test reads the contract's list of types and fails when one has no family.
 *
 * A row written by an older version can lack a field, and the log keeps what it was given, so each description reads
 * its fields as optional and says nothing for one that is missing.
 */
import {
  parseMessageOrigin,
  type HostEventDataMap,
  type InboundEventDataMap,
  type OutboundEventDataMap,
  type StoredEvent,
} from "@invisible-dots/shared/browser";
import { CHANNEL_LABELS, viaChannel } from "../channels";
import type { Tone } from "../tone";
import { toolLabel } from "./tool-labels";

/** What the Activity page filters by: one family per kind of thing that happens to a Dot. */
export const EVENT_FAMILIES = ["chat", "tasks", "tools", "approvals", "memory", "browser", "computer", "channels", "dot"] as const;
export type EventFamily = (typeof EVENT_FAMILIES)[number];

export const FAMILY_LABELS: Record<EventFamily, string> = {
  chat: "Chat",
  tasks: "Tasks",
  tools: "Tools",
  approvals: "Approvals",
  memory: "Memory",
  browser: "Browser",
  computer: "Computer",
  channels: "Channels",
  dot: "Dot",
};

/** The data of every type the log holds: the guest's events, the host's, and the person's own message, which the host stores. */
type EventData = OutboundEventDataMap & HostEventDataMap & { "user.message": Pick<InboundEventDataMap["user.message"], "text"> & { message_id: string; origin?: unknown } };
type KnownType = keyof EventData;

/** The family of each type. Exhaustive: a type added to the contract has no family until it is named here. */
const FAMILY_OF: Record<KnownType, EventFamily> = {
  "user.message": "chat",
  "message.assistant": "chat",
  "task.created": "tasks",
  "task.started": "tasks",
  "task.progress": "tasks",
  "task.completed": "tasks",
  "task.failed": "tasks",
  "task.cancelled": "tasks",
  "tool.called": "tools",
  "approval.requested": "approvals",
  "approval.resolved": "approvals",
  "memory.updated": "memory",
  "browser.identity.created": "browser",
  "browser.identity.deleted": "browser",
  "browser.identity.launched": "browser",
  "browser.identity.closed": "browser",
  "computer.state": "computer",
  "computer.started": "computer",
  "computer.stopped": "computer",
  "agent.started": "computer",
  "agent.state": "computer",
  "automation.next_run": "computer",
  "guest.event.refused": "computer",
  "channel.status": "channels",
  "channel.peer.paired": "channels",
  "channel.changed": "channels",
  "dot.created": "dot",
  "dot.updated": "dot",
  "dot.deleted": "dot",
};

/** The event types of a family, from the table above. */
export function typesOf(family: EventFamily): string[] {
  return (Object.keys(FAMILY_OF) as KnownType[]).filter((type) => FAMILY_OF[type] === family);
}

/** The family of a type, or null for one this version does not know. */
export function familyOf(type: string): EventFamily | null {
  return Object.hasOwn(FAMILY_OF, type) ? FAMILY_OF[type as KnownType] : null;
}

export interface EventView {
  id: number;
  type: string;
  /** Null for a type this version has no description of (it is shown as it is stored). */
  family: EventFamily | null;
  tone: Tone;
  title: string;
  /** One line: long text is cut, and the whole of it stays in the row's data. */
  detail: string;
  /** Where the message came from when it was not the web ("via Telegram"). */
  via: string | null;
  /** The tool a `tool.called` row is about, for its icon. */
  tool: string | null;
  source: StoredEvent["source"];
  at: string;
}

const MAX_DETAIL = 280;

/** One line of at most `max` characters: line breaks and runs of spaces become one space, a cut ends in "...". */
export function truncate(text: string, max = MAX_DETAIL): string {
  const flat = text.replace(/\s+/g, " ").trim();
  return flat.length > max ? `${flat.slice(0, max - 3)}...` : flat;
}

function text(value: unknown): string {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return "";
}

function joined(...parts: string[]): string {
  return parts.filter(Boolean).join(" ");
}

/** A channel as the person names it ("Telegram"); a kind this version does not know is shown as the host wrote it. */
function channelName(kind: unknown): string {
  return typeof kind === "string" && Object.hasOwn(CHANNEL_LABELS, kind) ? CHANNEL_LABELS[kind as keyof typeof CHANNEL_LABELS] : text(kind);
}

function identity(data: Partial<OutboundEventDataMap["browser.identity.created"]>): string {
  const name = text(data.name);
  const id = text(data.identity_id);
  return name && id ? `${name} (${id})` : name || id;
}

interface Draft {
  title: string;
  detail?: string;
  tone?: Tone;
  via?: string;
  tool?: string;
}

/** How a tool call ended, in the words of the row. A denied call never ran; an interrupted one has no known outcome. */
function outcome(data: Partial<OutboundEventDataMap["tool.called"]>): { word: string; tone: Tone } {
  if (data.interrupted === true) return { word: "interrupted", tone: "warn" };
  if (data.decision === "deny") return { word: "denied", tone: "error" };
  return data.ok === true ? { word: "ok", tone: "neutral" } : { word: "failed", tone: "error" };
}

/** What a task event says, after the task it is about. */
const withTask = (taskId: unknown, rest: string): string => joined(text(taskId) && `${text(taskId)}:`, rest);

const DESCRIBE: { [K in KnownType]: (data: Partial<EventData[K]>) => Draft } = {
  "user.message": (d) => {
    const via = viaChannel(parseMessageOrigin(d.origin) ?? undefined);
    return { title: "You sent a message", detail: text(d.text), ...(via ? { via } : {}) };
  },
  "message.assistant": (d) => ({ title: "Assistant replied", detail: text(d.text) }),
  "task.created": (d) => ({
    title: "Task created",
    detail: joined(text(d.description), typeof d.priority === "number" ? `(priority ${d.priority})` : ""),
  }),
  "task.started": (d) => ({ title: "Task started", detail: text(d.task_id) }),
  "task.progress": (d) => ({ title: "Task progress", detail: withTask(d.task_id, text(d.text)) }),
  "task.completed": (d) => ({ title: "Task completed", detail: withTask(d.task_id, text(d.summary)), tone: "ok" }),
  "task.failed": (d) => ({ title: "Task failed", detail: withTask(d.task_id, text(d.error)), tone: "error" }),
  "task.cancelled": (d) => ({ title: "Task cancelled", detail: text(d.task_id), tone: "neutral" }),
  "tool.called": (d) => {
    const { word, tone } = outcome(d);
    const duration = typeof d.duration_ms === "number" ? ` in ${Math.round(d.duration_ms)} ms` : "";
    const tool = text(d.tool);
    // What it acted on, how it ended, and which tool under which permission the policy decided on.
    const policy = joined(tool, text(d.permission) && `[${text(d.permission)}]`, text(d.decision));
    return {
      title: tool ? toolLabel(tool, d.tty === true).label : "Tool call",
      detail: [text(d.target), `${word}${duration}`, policy].filter(Boolean).join(" | "),
      tone,
      tool,
    };
  },
  "approval.requested": (d) => ({
    title: "Approval requested",
    detail: joined(text(d.tool), text(d.permission) && `[${text(d.permission)}]`, text(d.reason) && `- ${text(d.reason)}`),
    tone: "warn",
  }),
  "approval.resolved": (d) => ({
    title: d.decision === "approve" ? (d.always === true ? "Approved, and always allowed" : "Approved") : d.decision === "reject" ? "Rejected" : "Approval resolved",
    detail: joined(text(d.approval_id), text(d.note) && `- ${text(d.note)}`),
    tone: d.decision === "reject" ? "warn" : "ok",
  }),
  "browser.identity.created": (d) => ({ title: "Browser identity created", detail: identity(d), tone: "ok" }),
  "browser.identity.deleted": (d) => ({ title: "Browser identity deleted", detail: identity(d), tone: "neutral" }),
  "browser.identity.launched": (d) => ({ title: "Browser launched", detail: identity(d) }),
  "browser.identity.closed": (d) => ({ title: "Browser closed", detail: identity(d), tone: "neutral" }),
  "channel.status": (d) => ({
    title: "Channel status",
    detail: joined(channelName(d.kind), text(d.status), text(d.detail) && `- ${text(d.detail)}`),
    tone: d.status === "error" ? "error" : d.status === "needs_relink" ? "warn" : d.status === "connected" ? "ok" : "neutral",
  }),
  "channel.changed": (d) => ({ title: `Channel ${text(d.change) || "changed"}`, detail: channelName(d.kind), tone: d.change === "removed" ? "warn" : "neutral" }),
  "channel.peer.paired": (d) => ({ title: "Person paired", detail: joined(text(d.label), channelName(d.kind) && `on ${channelName(d.kind)}`), tone: "ok" }),
  "dot.created": (d) => ({ title: "Dot created", detail: text(d.name), tone: "ok" }),
  // Two things write it: a saved config (`name` only) and a Dot that went to ERROR (`status` and `error` too, lifecycle.ts).
  "dot.updated": (d) =>
    d.status === "ERROR"
      ? { title: "The Dot failed", detail: joined(text(d.name), text(d.error) && `- ${text(d.error)}`), tone: "error" }
      : { title: "Configuration updated", detail: text(d.name), tone: "neutral" },
  "dot.deleted": (d) => ({ title: "Dot deleted", detail: text(d.name), tone: "warn" }),
  "computer.state": (d) => ({ title: "Computer state", detail: text(d.state), tone: d.state === "ERROR" ? "error" : "neutral" }),
  "computer.started": () => ({ title: "Computer started", tone: "ok" }),
  "computer.stopped": () => ({ title: "Computer stopped", tone: "neutral" }),
  // The agent process came up: after a boot or a restart inside a running computer. It keeps the model key in memory
  // only, so the control plane sends it again when it sees this event.
  "agent.started": () => ({ title: "The engine started", detail: "The key was sent again", tone: "info" }),
  "agent.state": (d) => ({ title: "Agent state", detail: text(d.state), tone: "neutral" }),
  // The engine reports when its earliest enabled automation is next due (null: none is); the host wakes a stopped
  // computer shortly before that time.
  "automation.next_run": (d) => ({
    title: "Next automation",
    detail: typeof d.next_run_at_ms === "number" ? `due ${new Date(d.next_run_at_ms).toISOString()}` : "none due",
    tone: "neutral",
  }),
  // The Dot brought MEMORY.md, what it is given about the person in every prompt, up to date from its conversations.
  "memory.updated": (d) => ({
    title: d.changed === false ? "Memory checked, nothing new" : "Memory updated",
    detail:
      typeof d.conversations === "number" ? `from ${d.conversations} conversation${d.conversations === 1 ? "" : "s"}` : "",
    tone: d.changed === false ? "neutral" : "info",
  }),
  // The engine sent something the control plane could not read, so it is not in this log: this row is all there is of it.
  "guest.event.refused": (d) => ({
    title: "The computer sent an event that was not read",
    detail: joined(text(d.type), typeof d.seq === "number" ? `(#${d.seq})` : "", text(d.problem) && `- ${text(d.problem)}`),
    tone: "warn",
  }),
};

function describeKnown<K extends KnownType>(type: K, data: Record<string, unknown>): Draft {
  return (DESCRIBE[type] as (data: Partial<EventData[K]>) => Draft)(data as Partial<EventData[K]>);
}

/** The row of the Activity page for one stored event. */
export function viewEvent(event: StoredEvent): EventView {
  const type: string = event.type;
  const data = event.data ?? {};
  const draft: Draft = Object.hasOwn(DESCRIBE, type) ? describeKnown(type as KnownType, data) : { title: type, detail: JSON.stringify(data), tone: "neutral" };
  return {
    id: event.id,
    type,
    family: familyOf(type),
    tone: draft.tone ?? "info",
    title: draft.title,
    detail: truncate(draft.detail ?? ""),
    via: draft.via ?? null,
    tool: draft.tool ?? null,
    source: event.source,
    at: event.created_at,
  };
}
