"use client";

import { USAGE_EVENT_TYPES } from "@invisible-dots/shared/browser";
import { api } from "../../lib/api";
import { formatUsd, startOfToday } from "../../lib/format";
import { DotEventScope, useLiveRefresh } from "../events";
import { useResource } from "../ui";


/**
 * What the Dot's model calls cost since midnight, as its guest reported them. It is scoped to its own Dot, so
 * wherever it is shown (the Dot's header, every card of Home) it re-reads only when that Dot spends.
 */
export function CostPill({ dotId }: { dotId: string }) {
  return (
    <DotEventScope dotId={dotId}>
      <Spend dotId={dotId} />
    </DotEventScope>
  );
}

function Spend({ dotId }: { dotId: string }) {
  const since = startOfToday();
  const usage = useResource(() => api.usage(dotId, { since }), `usage:${dotId}:${since}`);
  useLiveRefresh(usage.reload, USAGE_EVENT_TYPES);
  if (usage.data === undefined) return null;
  return (
    <span className="font-mono text-xs text-muted-foreground tabular-nums" title="Model spend today, as the Dot's computer reported it">
      <span className="sr-only">Spent today: </span>
      {formatUsd(usage.data.spent_usd)}
    </span>
  );
}
