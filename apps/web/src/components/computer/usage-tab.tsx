"use client";

import { AlertCircleIcon, PlayIcon, RefreshCwIcon, RotateCwIcon, SquareIcon } from "lucide-react";
import { computerIsUp, USAGE_EVENT_TYPES } from "@invisible-dots/shared/browser";
import type { ReactNode } from "react";
import { api } from "../../lib/api";
import { allowedActions, computerView, taskRunning, type Usage } from "../../lib/computer";
import { formatBytes, formatDate, formatDuration, formatUsd, startOfToday } from "../../lib/format";
import { cn } from "../../lib/utils";
import { useDot } from "../DotShell";
import { ErrorAlert } from "../ErrorAlert";
import { usePower } from "../dot/use-power";
import { useLiveRefresh } from "../events";
import { useResource } from "../ui";
import { Alert, AlertDescription, AlertTitle } from "../ui/alert";
import { Button } from "../ui/button";
import { Skeleton } from "../ui/skeleton";
import { AutomationsNote } from "./automations-note";
import { ComputerStatus } from "./computer-status";

const COMPUTER_EVENTS = ["computer.state", "computer.started", "computer.stopped"];

/** A bar of how much of a resource is used: amber from three quarters, red from nine tenths. */
function Meter({ label, usage }: { label: string; usage: Usage }) {
  const percent = Math.round(usage.fraction * 100);
  return (
    <div className="space-y-1">
      <div className="flex flex-wrap justify-between gap-x-4 text-sm">
        <span>{label}</span>
        <span className="text-muted-foreground">
          {formatBytes(usage.usedBytes)} of {formatBytes(usage.totalBytes)} used ({percent}%)
        </span>
      </div>
      <div role="meter" aria-label={`${label} used`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent} aria-valuetext={`${percent}%`} className="h-2 overflow-hidden rounded-full bg-muted">
        <div className={cn("h-full rounded-full", usage.fraction >= 0.9 ? "bg-danger" : usage.fraction >= 0.75 ? "bg-warn" : "bg-ok")} style={{ width: `${percent}%` }} />
      </div>
    </div>
  );
}

function Facts({ children }: { children: ReactNode }) {
  return <dl className="grid grid-cols-[max-content_minmax(0,1fr)] gap-x-6 gap-y-1.5 text-sm [&_dd]:min-w-0 [&_dd]:break-words [&_dt]:text-muted-foreground">{children}</dl>;
}

function Card({ id, title, children }: { id: string; title: string; children: ReactNode }) {
  return (
    <section aria-labelledby={id} className="space-y-3 rounded-lg border bg-card p-4 text-card-foreground">
      <h3 id={id} className="text-sm font-semibold">
        {title}
      </h3>
      {children}
    </section>
  );
}

/** What the model has cost: today, and since the Dot began, as the Dot's computer reported it. */
function Spend({ dotId }: { dotId: string }) {
  const since = startOfToday();
  const today = useResource(() => api.usage(dotId, { since }), `usage-today:${dotId}:${since}`);
  const total = useResource(() => api.usage(dotId), `usage-total:${dotId}`);
  useLiveRefresh(() => {
    today.reload();
    total.reload();
  }, USAGE_EVENT_TYPES);
  return (
    <Card id="usage-spend" title="Model spend">
      <ErrorAlert error={today.error ?? total.error} title="Could not read the spend" />
      <Facts>
        <dt>Today</dt>
        <dd>{today.data === undefined ? "-" : formatUsd(today.data.spent_usd)}</dd>
        <dt>In total</dt>
        <dd>{total.data === undefined ? "-" : formatUsd(total.data.spent_usd)}</dd>
      </Facts>
      <p className="text-xs text-muted-foreground">What the Dot&apos;s computer reported for answers and finished tasks, in US dollars as OpenRouter priced them. Tokens are not reported, only the cost. Work that was cancelled or cut short is not counted.</p>
    </Card>
  );
}

/**
 * The Dot's computer in numbers: what it was given, what it uses now while it runs, the images it started from, and
 * what the Dot's model has cost; and the buttons that start, reboot and stop it. It reads while the computer is off
 * too: a computer that failed to start says why here.
 */
export function UsageTab() {
  const { dotId, dot } = useDot();
  const computer = useResource(() => api.computer(dotId), `computer:${dotId}`);
  useLiveRefresh(computer.reload, COMPUTER_EVENTS);
  const power = usePower({
    dotId,
    taskRunning: dot.data !== undefined && taskRunning(dot.data.status),
    onDone: () => {
      computer.reload();
      dot.reload();
    },
  });

  const view = computerView(computer.data ?? null, dot.data?.config ?? null);
  const allowed = allowedActions(view.state);
  const up = computerIsUp(view.state);
  const idle = view.allocated.idleTimeout;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <ComputerStatus state={view.state} />
        <div className="ml-auto flex flex-wrap gap-2">
          <Button type="button" variant="outline" size="sm" onClick={computer.reload} disabled={computer.loading}>
            <RefreshCwIcon className={cn(computer.loading && "animate-spin motion-reduce:animate-none")} />
            Refresh
          </Button>
          <Button type="button" variant="outline" size="sm" disabled={power.pending || !allowed.start} onClick={() => void power.act("start")}>
            <PlayIcon />
            Start
          </Button>
          <Button type="button" variant="outline" size="sm" disabled={power.pending || !allowed.reboot} onClick={() => void power.act("reboot")}>
            <RotateCwIcon />
            Reboot
          </Button>
          <Button type="button" variant="outline" size="sm" disabled={power.pending || !allowed.stop} onClick={() => void power.act("stop")}>
            <SquareIcon />
            Stop
          </Button>
        </div>
      </div>

      <ErrorAlert error={computer.error} title="Could not load the computer" />
      {computer.data?.last_error ? (
        <Alert variant="destructive">
          <AlertCircleIcon />
          <AlertTitle>The last start or stop failed</AlertTitle>
          <AlertDescription>
            <p>{computer.data.last_error}</p>
          </AlertDescription>
        </Alert>
      ) : null}
      {computer.data === undefined && !computer.error ? <Skeleton className="h-40 w-full" aria-busy="true" /> : null}

      {computer.data !== undefined ? (
        <div className="grid gap-4 md:grid-cols-2">
          <Card id="usage-allocated" title="Given to the computer">
            <Facts>
              <dt>vCPUs</dt>
              <dd>{view.allocated.cpus ?? "-"}</dd>
              <dt>Memory</dt>
              <dd>{view.allocated.memory ?? "-"}</dd>
              <dt>Disk</dt>
              <dd>{view.allocated.disk ?? "-"}</dd>
              <dt>Sleeps after idle</dt>
              <dd>{idle && /^0+[a-z]*$/.test(idle) ? "never" : (idle ?? "-")}</dd>
              <dt>Last active</dt>
              <dd>{computer.data.last_active_at ? formatDate(computer.data.last_active_at) : "-"}</dd>
            </Facts>
          </Card>

          <Card id="usage-live" title="In use now">
            {view.live ? (
              <>
                <Facts>
                  <dt>Hostname</dt>
                  <dd>{view.live.hostname || "-"}</dd>
                  <dt>Uptime</dt>
                  <dd>{formatDuration(view.live.uptimeSeconds)}</dd>
                  <dt>CPUs</dt>
                  <dd>{view.live.cpus}</dd>
                </Facts>
                <Meter label="Memory" usage={view.live.memory} />
                <Meter label="Disk" usage={view.live.disk} />
              </>
            ) : (
              <p className="text-sm text-muted-foreground">{up ? "The computer did not report its usage." : "What it uses is shown while the computer runs."}</p>
            )}
          </Card>

          <Card id="usage-images" title="Images">
            <Facts>
              <dt>Ready</dt>
              <dd>{computer.data.ready ? "Yes: the Dot has its key and answers" : "Not yet"}</dd>
              <dt>Golden image</dt>
              <dd>{computer.data.golden_image ? <code className="text-xs">{computer.data.golden_image}</code> : "-"}</dd>
              <dt>Runtime image</dt>
              <dd>{computer.data.runtime_image ? <code className="text-xs">{computer.data.runtime_image}</code> : "-"}</dd>
            </Facts>
          </Card>

          <Card id="usage-automations" title="Automations">
            <AutomationsNote dotId={dotId} />
          </Card>

          <Spend dotId={dotId} />
        </div>
      ) : null}
    </div>
  );
}
