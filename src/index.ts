#!/usr/bin/env node
import * as acp from "@agentclientprotocol/sdk";
import { Readable, Writable } from "node:stream";
import { ZenAgent } from "./agent.js";

const agent = new ZenAgent();

const output = Writable.toWeb(process.stdout);
const input = Readable.toWeb(process.stdin) as ReadableStream<Uint8Array>;
const stream = acp.ndJsonStream(output, input);

acp
  .agent({ name: "zen-agent" })
  .onRequest("initialize", (ctx) => agent.initialize(ctx.params))
  .onRequest("authenticate", (ctx) => agent.authenticate(ctx.params))
  .onRequest("session/new", (ctx) => agent.newSession(ctx.params))
  .onRequest("session/load", (ctx) => agent.loadSession(ctx.params, ctx.client))
  .onRequest("session/list", (ctx) => agent.listSessions(ctx.params))
  .onRequest("session/delete", (ctx) => agent.deleteSession(ctx.params))
  .onRequest("session/resume", (ctx) => agent.resumeSession(ctx.params))
  .onRequest("session/close", (ctx) => agent.closeSession(ctx.params))
  .onRequest("session/prompt", (ctx) => agent.prompt(ctx.params, ctx.client))
  .onNotification("session/cancel", (ctx) => agent.cancel(ctx.params))
  .connect(stream);
