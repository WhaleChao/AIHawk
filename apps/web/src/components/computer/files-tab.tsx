"use client";

import { ApiError } from "@invisible-dots/sdk";
import type { FileEntry } from "@invisible-dots/shared/browser";
import { FileIcon, FolderIcon, RefreshCwIcon } from "lucide-react";
import Link from "next/link";
import { useMemo } from "react";
import { api } from "../../lib/api";
import { isComputerStopped } from "../../lib/computer";
import { computerHref, type ComputerQuery } from "../../lib/computer-view";
import { breadcrumbs, childPath, sortEntries } from "../../lib/files";
import { formatBytes, formatDate } from "../../lib/format";
import { relativeTime } from "../../lib/time";
import { cn } from "../../lib/utils";
import { ErrorAlert } from "../ErrorAlert";
import { useLiveRefresh } from "../events";
import { useResource } from "../ui";
import { Button } from "../ui/button";
import { Skeleton } from "../ui/skeleton";
import { ComputerOff } from "./computer-off";
import { FilePreview } from "./file-preview";

/** What the Dot does that adds or changes files: the page reads the folder again after each. */
const FILE_EVENTS = ["task.completed", "task.failed", "message.assistant", "memory.updated"];

/** The sentence for a folder that could not be listed, for the answers a person can do something about. */
function listProblem(error: unknown): string | null {
  if (error instanceof ApiError) {
    if (error.code === "not_found") return "This folder does not exist (any more).";
    if (error.code === "not_a_directory") return "This is a file, not a folder.";
    if (error.code === "outside_home") return "This path leads outside the Dot's home folder. Only the home folder is shown.";
  }
  return null;
}

/**
 * The files of the Dot's home folder, read only: folders to walk through, and a file's text or picture. The address
 * holds the folder and the file, so it can be linked and the back button walks back up. Reading needs the computer
 * running.
 */
export function FilesTab({ dotId, query }: { dotId: string; query: ComputerQuery }) {
  const listing = useResource(() => api.listFiles(dotId, query.path ?? undefined), `files:${dotId}:${query.path ?? ""}`);
  useLiveRefresh(listing.reload, FILE_EVENTS);
  const failure = listing.error;
  const folder = listing.data?.path ?? null;
  const entries = useMemo(() => sortEntries(listing.data?.entries ?? []), [listing.data]);
  const open = query.file === null ? null : (entries.find((entry) => entry.name === query.file && entry.type === "file") ?? null);
  const href = (path: string, file?: string) => computerHref(dotId, { view: "files", path, ...(file ? { file } : {}) });

  // The Computer page explains a computer it knows is off; this is the same answer from a computer that stopped meanwhile.
  if (isComputerStopped(failure)) return <ComputerOff dotId={dotId} state="STOPPED" what="Start the computer to see its files" />;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <nav aria-label="Folder" className="min-w-0 flex-1">
          <ol className="flex flex-wrap items-center gap-1 text-sm">
            {(folder === null ? [] : breadcrumbs(folder)).map((crumb, index, all) => (
              <li key={crumb.path} className="flex items-center gap-1">
                {index > 0 ? (
                  <span aria-hidden="true" className="text-muted-foreground">
                    /
                  </span>
                ) : null}
                {index === all.length - 1 && query.file === null ? (
                  <span aria-current="page" className="font-medium">
                    {crumb.name}
                  </span>
                ) : (
                  <Link href={href(crumb.path)} className="text-muted-foreground hover:text-foreground hover:underline">
                    {crumb.name}
                  </Link>
                )}
              </li>
            ))}
          </ol>
        </nav>
        <Button type="button" variant="outline" size="sm" onClick={listing.reload} disabled={listing.loading}>
          <RefreshCwIcon className={cn(listing.loading && "animate-spin motion-reduce:animate-none")} />
          Refresh
        </Button>
      </div>

      {failure ? (
        listProblem(failure) ? (
          <div role="alert" className="space-y-2 rounded-lg border border-dashed p-6 text-center">
            <p className="text-sm">{listProblem(failure)}</p>
            <Button asChild variant="outline" size="sm">
              <Link href={computerHref(dotId, { view: "files" })}>Go to the home folder</Link>
            </Button>
          </div>
        ) : (
          <ErrorAlert error={failure} title="Could not list the folder" />
        )
      ) : null}

      {listing.data === undefined && !failure ? (
        <div className="space-y-2" aria-busy="true">
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-8 w-full" />
          <Skeleton className="h-8 w-full" />
        </div>
      ) : null}

      {listing.data !== undefined && folder !== null ? (
        entries.length === 0 ? (
          <p className="rounded-lg border border-dashed p-8 text-center text-sm text-muted-foreground">This folder is empty.</p>
        ) : (
          <div className="overflow-x-auto rounded-lg border">
            <table className="w-full text-sm">
              <thead className="bg-muted text-left text-xs text-muted-foreground">
                <tr>
                  <th scope="col" className="px-3 py-2 font-medium">
                    Name
                  </th>
                  <th scope="col" className="px-3 py-2 text-right font-medium">
                    Size
                  </th>
                  <th scope="col" className="px-3 py-2 font-medium">
                    Modified
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y">
                {entries.map((entry) => (
                  <Row key={entry.name} entry={entry} folder={folder} current={entry.name === open?.name} href={href} />
                ))}
              </tbody>
            </table>
          </div>
        )
      ) : null}

      {query.file !== null && listing.data !== undefined && folder !== null ? (
        open !== null ? (
          <FilePreview key={`${folder}/${open.name}`} dotId={dotId} folder={folder} entry={open} />
        ) : (
          <p role="status" className="text-sm text-muted-foreground">
            There is no file called {query.file} in this folder (any more).
          </p>
        )
      ) : null}
    </div>
  );
}

function Row({ entry, folder, current, href }: { entry: FileEntry; folder: string; current: boolean; href: (path: string, file?: string) => string }) {
  const Icon = entry.type === "dir" ? FolderIcon : FileIcon;
  const label =
    entry.type === "dir" ? (
      <Link href={href(childPath(folder, entry.name))} className="hover:underline">
        {entry.name}
        <span className="sr-only"> (folder)</span>
      </Link>
    ) : entry.type === "file" ? (
      <Link href={href(folder, entry.name)} aria-current={current ? "true" : undefined} className="hover:underline">
        {entry.name}
      </Link>
    ) : (
      // A link, a socket, a device: nothing a person can open here.
      <span title="Not a file or a folder">{entry.name}</span>
    );
  return (
    <tr className={cn(current && "bg-accent")}>
      <td className="px-3 py-1.5">
        <span className="flex items-center gap-2">
          <Icon aria-hidden="true" className="size-4 shrink-0 text-muted-foreground" />
          <span className="min-w-0 break-all">{label}</span>
        </span>
      </td>
      <td className="px-3 py-1.5 text-right whitespace-nowrap text-muted-foreground">{entry.type === "file" ? formatBytes(entry.size) : "-"}</td>
      <td className="px-3 py-1.5 whitespace-nowrap text-muted-foreground">
        <time dateTime={entry.mtime} title={formatDate(entry.mtime)}>
          {relativeTime(entry.mtime)}
        </time>
      </td>
    </tr>
  );
}
