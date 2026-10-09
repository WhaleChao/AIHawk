"use client";

import { BrainIcon, CircleDotIcon, FlagIcon, GlobeIcon, HandIcon, MessageSquareIcon, MonitorIcon, SendIcon, SparklesIcon, type LucideIcon } from "lucide-react";
import { toolLabel } from "../../lib/events/tool-labels";
import type { EventFamily, EventView } from "../../lib/events/view";
import { formatDate } from "../../lib/format";
import type { StoredEvent } from "../../lib/types";
import { cn } from "../../lib/utils";
import { TONE_EDGE } from "../dot/tone";
import { FAMILY_ICON } from "../tool-family-icon";
import { Badge } from "../ui/badge";

const ICON: Record<EventFamily, LucideIcon> = {
  chat: MessageSquareIcon,
  tasks: FlagIcon,
  // A tool call draws the icon of its own kind of tool; this one is for a call with no tool named.
  tools: CircleDotIcon,
  approvals: HandIcon,
  memory: BrainIcon,
  browser: GlobeIcon,
  computer: MonitorIcon,
  channels: SendIcon,
  dot: SparklesIcon,
};

function iconOf(view: EventView): LucideIcon {
  if (view.tool) return FAMILY_ICON[toolLabel(view.tool).family];
  return view.family === null ? CircleDotIcon : ICON[view.family];
}

/**
 * One event of the log: what it was in words, where it came from, when, and under "Data" the event exactly as the
 * control plane stored it (the line is cut at 280 characters; the data is not).
 */
export function ActivityRow({ view, event }: { view: EventView; event: StoredEvent }) {
  const Icon = iconOf(view);
  return (
    <li data-testid="activity-row" data-tone={view.tone} className={cn("flex gap-3 rounded-md border border-l-4 bg-card px-3 py-2 text-card-foreground", TONE_EDGE[view.tone])}>
      <Icon aria-hidden="true" className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
      <div className="min-w-0 flex-1 space-y-1">
        <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
          <span className="text-sm font-medium">{view.title}</span>
          {view.via ? <Badge variant="outline">{view.via}</Badge> : null}
          <code className="text-xs text-muted-foreground">{view.type}</code>
          <span className="text-xs text-muted-foreground">{view.source}</span>
        </div>
        {view.detail ? <p className="text-sm break-words text-muted-foreground">{view.detail}</p> : null}
        <details className="text-xs">
          <summary className="w-fit cursor-pointer rounded-sm text-muted-foreground hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-hidden">Data</summary>
          <pre tabIndex={0} aria-label={`Data of event ${view.id}`} className="mt-1 max-h-64 overflow-auto rounded-md bg-muted p-2 font-mono break-words whitespace-pre-wrap">
            {JSON.stringify(event.data, null, 2)}
          </pre>
        </details>
      </div>
      <time dateTime={view.at} className="shrink-0 text-xs text-muted-foreground">
        {formatDate(view.at)}
      </time>
    </li>
  );
}
