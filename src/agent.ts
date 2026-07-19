import OpenAI from "openai";
import type { ChatCompletionMessageParam } from "openai/resources/chat/completions";
import type { AgentConfig } from "./config.ts";
import { allTools, findTool, isDangerous, getToolsWithoutDelegate } from "./tools.ts";
import type { Tool } from "./tools.ts";

// ============================================================
// 颜色输出辅助函数 —— 让终端输出更直观地展示 Agent 内部运转
// ============================================================
const c = {
  dim: (s: string) => `\x1b[2m${s}\x1b[0m`,
  cyan: (s: string) => `\x1b[36m${s}\x1b[0m`,
  yellow: (s: string) => `\x1b[33m${s}\x1b[0m`,
  green: (s: string) => `\x1b[32m${s}\x1b[0m`,
  blue: (s: string) => `\x1b[34m${s}\x1b[0m`,
  red: (s: string) => `\x1b[31m${s}\x1b[0m`,
  bold: (s: string) => `\x1b[1m${s}\x1b[0m`,
  magenta: (s: string) => `\x1b[35m${s}\x1b[0m`,
};

/**
 * 确认回调函数类型
 * 接收工具名和参数，返回用户是否同意执行
 * 用于危险操作（file_write、run_bash）执行前的用户确认
 */
export type ConfirmCallback = (
  toolName: string,
  args: Record<string, unknown>,
) => Promise<boolean>;

/**
 * 流式响应的工具调用累积结构
 * 因为流式传输中 tool_calls 的 id、name、arguments 是分多个 chunk 到达的，
 * 需要手动拼装成完整的工具调用对象
 */
interface AccumulatedToolCall {
  id: string;
  name: string;
  arguments: string;
}

// ============================================================
// 上下文管理 —— 防止 messages 数组无限膨胀
// ============================================================

// 上下文上限（字符数），约等于 64000 tokens
// 超出后自动裁剪旧消息，保留 system prompt + 最近的消息
const MAX_CONTEXT_CHARS = 128000;
// 裁剪后至少保留最近的消息条数（防止把刚发生的对话也裁掉）
const MIN_KEEP_MESSAGES = 6;

/**
 * 估算 messages 数组的总字符数
 * 把每条消息的 content 和 tool_calls 的字符数累加
 */
function estimateContextSize(messages: ChatCompletionMessageParam[]): number {
  let total = 0;
  for (const msg of messages) {
    if (typeof msg.content === "string") {
      total += msg.content.length;
    }
    total += msg.role.length;
    if ("tool_calls" in msg && msg.tool_calls) {
      for (const tc of msg.tool_calls) {
        total += tc.function.name.length + tc.function.arguments.length;
      }
    }
  }
  return total;
}

/**
 * 裁剪 messages 数组，保留 system prompt 和最近的消息
 *
 * 策略：
 *   1. system prompt（索引 0）永远保留
 *   2. 从索引 1 开始，删除最旧的消息，直到总字符数降到阈值以下
 *   3. 至少保留 MIN_KEEP_MESSAGES 条消息（不含 system prompt）
 *
 * 这种"滑动窗口"策略是最简单的上下文管理方式，
 * 缺点是会丢失早期的对话信息。
 * 更高级的做法（如摘要压缩）见下方注释。
 *
 * @returns 被裁剪掉的消息数量，0 表示不需要裁剪
 */
export function compactMessages(messages: ChatCompletionMessageParam[]): number {
  if (messages.length <= 1) return 0;

  const size = estimateContextSize(messages);
  if (size <= MAX_CONTEXT_CHARS) return 0;

  const removed = messages.length - MIN_KEEP_MESSAGES - 1;
  if (removed <= 0) return 0;

  console.log(
    c.yellow(
      `\n⚠️ 上下文超限 (${size} 字符 > ${MAX_CONTEXT_CHARS} 字符)，裁剪 ${removed} 条旧消息`,
    ),
  );

  messages.splice(1, removed);

  const newSize = estimateContextSize(messages);
  console.log(c.dim(`   裁剪后: ${newSize} 字符，${messages.length} 条消息`));

  return removed;
}

// ============================================================
// 反思/纠错 —— 工具调用失败后 LLM 自动分析原因并重试
// ============================================================

// 连续失败多少次后强制停止，防止无限重试
const MAX_CONSECUTIVE_FAILURES = 3;

/**
 * 判断工具返回结果是否为错误
 * 通过关键词匹配识别：中文错误前缀、HTTP 错误码、异常信息等
 */
function isToolError(result: string): boolean {
  const errorPatterns = [
    "错误",
    "无法",
    "失败",
    "[ERROR]",
    "操作被用户拒绝",
    "Error:",
    "not found",
    "HTTP 4",
    "HTTP 5",
  ];
  return errorPatterns.some((p) => result.includes(p));
}

/**
 * 标准化错误格式，加上 [ERROR] 前缀方便 LLM 识别
 */
function formatToolResult(result: string): { output: string; isError: boolean } {
  const err = isToolError(result);
  return {
    output: err ? `[ERROR] ${result}` : `[SUCCESS] ${result}`,
    isError: err,
  };
}

// ============================================================
// 多 Agent 协作 —— 委派子 Agent 独立执行任务
// ============================================================

/**
 * 运行一个子 Agent 循环
 *
 * 子 Agent 拥有独立的 messages 上下文，不会污染主 Agent 的对话历史。
 * 子 Agent 使用简化的工具列表（不含 delegate，防止递归），
 * 且运行轮数更少（10 轮），避免子 Agent 消耗过多时间。
 *
 * @param task - 主 Agent 为子 Agent 编写的系统提示词
 * @param context - 补充上下文信息
 * @param allowedTools - 子 Agent 可用的工具名称列表
 * @returns 子 Agent 的最终回复，作为工具结果返回给主 Agent
 */
async function runSubAgent(
  client: OpenAI,
  config: AgentConfig,
  task: string,
  context: string,
  allowedTools: string[],
  confirmCallback: ConfirmCallback,
  label: string,
): Promise<string> {
  // 子 Agent 的独立对话上下文
  const subMessages: ChatCompletionMessageParam[] = [
    {
      role: "system",
      content: task + "\n\n你的回复将作为子任务结果返回给主 Agent。请简洁、准确地完成任务。",
    },
    {
      role: "user",
      content: context || "开始执行任务",
    },
  ];

  // 子 Agent 可用的工具：排除 delegate，防止无限递归
  let subTools: Tool[] = getToolsWithoutDelegate();
  if (allowedTools.length > 0) {
    // 如果主 Agent 指定了工具列表，进一步过滤
    subTools = subTools.filter((t) => allowedTools.includes(t.definition.name));
  }

  console.log(c.magenta("\n  ┌──────────────────────────────────────┐"));
  console.log(c.magenta(`  │  🤖 ${label} 启动                      │`));
  console.log(c.magenta(`  │  任务: ${task.slice(0, 40)}...  │`));
  console.log(c.magenta(`  │  可用工具: ${subTools.map((t) => t.definition.name).join(", ")}`.padEnd(42) + `│`));
  console.log(c.magenta("  └──────────────────────────────────────┘"));

  const result = await runAgentLoop(
    client,
    config,
    subMessages,
    confirmCallback,
    { logPrefix: "  │ ", tools: subTools, maxTurns: 10, label, noStream: true },
  );

  console.log(c.magenta("  ┌──────────────────────────────────────┐"));
  console.log(c.magenta(`  │  ✅ ${label} 完成                      │`));
  console.log(c.magenta("  └──────────────────────────────────────┘"));

  return result;
}

// ============================================================
// Agent 循环
// ============================================================

export interface RunAgentLoopOptions {
  /** 输出日志的前缀，子 Agent 使用 "  │ " 实现缩进效果 */
  logPrefix?: string;
  /** 使用的工具列表，子 Agent 不能包含 delegate */
  tools?: Tool[];
  /** 最大循环轮数，子 Agent 比主 Agent 少 */
  maxTurns?: number;
  /** Agent 标签，用于区分主 Agent 和子 Agent */
  label?: string;
  /** 禁用流式输出，子 Agent 设为 true 避免并行输出穿插 */
  noStream?: boolean;
}

/**
 * ===== Agent 循环（流式输出 + 共享消息历史 + 多 Agent 协作） =====
 *
 * 这是 Agent 的核心：一个"思考-行动-观察"的循环。
 *
 * 流程：
 *   1. 使用传入的 messages 数组（已包含 system + 历史 + 新 user 消息）调用 LLM
 *   2. 实时流式接收 LLM 的响应，逐 token 打印到终端
 *   3. 流结束后判断：
 *      a) 文本回复 → 把 assistant 消息追加到 messages，退出循环
 *      b) 工具调用 → 进入步骤 4
 *   4. 如果是 delegate 工具 → 创建子 Agent 独立运行（可并行多个）
 *      如果是普通工具 → 逐个检查确认后执行
 *   5. 执行工具，把 assistant(tool_calls) 和 tool(result) 追加到 messages
 *   6. 回到步骤 1，让 LLM 根据新信息继续思考
 *
 * 这个循环会一直运行，直到 LLM 决定不再调用工具（给出最终回复）。
 */
export async function runAgentLoop(
  client: OpenAI,
  config: AgentConfig,
  messages: ChatCompletionMessageParam[],
  confirmCallback: ConfirmCallback,
  options: RunAgentLoopOptions = {},
): Promise<string> {
  const logPrefix = options.logPrefix ?? "";
  const tools = options.tools ?? allTools;
  const MAX_TURNS = options.maxTurns ?? 30;
  const label = options.label ?? "主 Agent";
  const noStream = options.noStream ?? false;

  // 带前缀的日志辅助函数
  const log = (msg: string) => console.log(logPrefix + msg);
  // 带前缀的流式输出
  const write = (s: string) => process.stdout.write(logPrefix + s);

  // 连续失败计数 —— 同一轮对话中工具调用连续失败的次数
  let consecutiveFailures = 0;

  for (let turn = 0; turn < MAX_TURNS; turn++) {
    log(c.dim(`\n--- [${label}] 循环 第 ${turn + 1} 轮 ---`));

    // 步骤 0: 上下文管理 —— 裁剪过长的对话历史
    compactMessages(messages);

    // =========================================
    // 步骤 1: 流式调用 LLM
    // =========================================
    log(c.cyan("📡 发送请求到 LLM（流式模式）..."));
    log(c.dim(`   模型: ${config.model}`));
    log(c.dim(`   当前对话消息数: ${messages.length}`));

    const stream = await client.chat.completions.create({
      model: config.model,
      messages,
      tools: tools.map((t) => ({
        type: "function" as const,
        function: t.definition,
      })),
      tool_choice: "auto",
      stream: true,
    });

    // ---- 累积器：从流式 chunk 中拼装完整的响应 ----
    let fullContent = "";
    const toolCallAcc = new Map<number, AccumulatedToolCall>();

if (!noStream) {
      write(c.blue("🤖 "));
    }

    for await (const chunk of stream) {
      const delta = chunk.choices?.[0]?.delta;
      if (!delta) continue;

      if (delta.content) {
        fullContent += delta.content;
        if (!noStream) {
          write(delta.content); // 主 Agent 逐 token 流式输出
        }
      }

      if (delta.tool_calls) {
        for (const tc of delta.tool_calls) {
          const idx = tc.index;
          if (!toolCallAcc.has(idx)) {
            toolCallAcc.set(idx, { id: "", name: "", arguments: "" });
          }
          const acc = toolCallAcc.get(idx)!;
          if (tc.id) acc.id = tc.id;
          if (tc.function?.name) acc.name += tc.function.name;
          if (tc.function?.arguments) acc.arguments += tc.function.arguments;
        }
      }
    }

    if (fullContent) {
      if (noStream) {
        write(c.blue("🤖 ") + fullContent + "\n");
      } else {
        write("\n");
      }
    }

    // ---- 判断流结束后的响应类型 ----

    /**
     * 情况 A: LLM 决定调用工具
     */
    if (toolCallAcc.size > 0) {
      log(c.yellow(`\n🔧 LLM 决定调用 ${toolCallAcc.size} 个工具:`));

      const toolCalls = Array.from(toolCallAcc.entries())
        .sort(([a], [b]) => a - b)
        .map(([_, tc]) => ({
          id: tc.id,
          type: "function" as const,
          function: {
            name: tc.name,
            arguments: tc.arguments,
          },
        }));

      messages.push({
        role: "assistant",
        content: fullContent || null,
        tool_calls: toolCalls,
      });

      // 收集同一轮中所有的 delegate 子任务，实现并行执行
      const delegatePromises: Array<Promise<{ id: string; result: string }>> = [];
      // 子 Agent 编号，用于输出标识
      let subAgentIndex = 0;

      for (const toolCall of toolCalls) {
        const toolName = toolCall.function.name;
        const toolArgs = JSON.parse(toolCall.function.arguments);
        const tool = findTool(toolName);

        log(c.yellow(`\n  📞 工具调用链:  ${toolName}`));
        log(c.yellow(`     ├─ ID:        ${toolCall.id}`));
        log(c.yellow(`     ├─ 参数:      ${JSON.stringify(toolArgs)}`));

        let toolResult: string;

        if (!tool) {
          toolResult = `[ERROR] 错误: 未找到工具 "${toolName}"`;
          log(c.red(`     └─ 返回结果:  ❌ ${toolResult}`));
          consecutiveFailures++;
          messages.push({ role: "tool", tool_call_id: toolCall.id, content: toolResult });
          if (consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
            log(c.red(`\n  🛑 连续失败 ${consecutiveFailures} 次，强制停止重试`));
            messages.push({
              role: "system",
              content: `⚠️ 已连续失败 ${consecutiveFailures} 次。请停止尝试，向用户解释失败原因。不要再次调用工具。`,
            });
          }
          continue;
        }

        const riskLabel = isDangerous(tool) ? "🔴 危险" : "🟢 安全";
        log(c.yellow(`     ├─ 风险等级:  ${riskLabel}`));

        /**
         * ===== 特殊处理：delegate 工具 =====
         * delegate 不通过 tool.execute() 执行，而是创建子 Agent 循环。
         * 多个 delegate 可以并行执行（Promise.all）。
         */
        if (toolName === "delegate") {
          const task = toolArgs["task"] as string;
          const context = (toolArgs["context"] as string) || "";
          const allowedTools = (toolArgs["tools"] as string)
            ? (toolArgs["tools"] as string).split(",").map((s) => s.trim())
            : [];

          log(c.yellow(`     ├─ 状态:      🚀 启动子 Agent...`));
          subAgentIndex++;
          const subLabel = `子 Agent ${subAgentIndex}`;
          delegatePromises.push(
            runSubAgent(client, config, task, context, allowedTools, confirmCallback, subLabel).then(
              (result) => ({ id: toolCall.id, result }),
            ),
          );
          continue; // 跳过当前循环，等待所有 delegate 完成后统一处理
        }

        // ---- 普通工具：权限检查 + 执行 ----
        if (isDangerous(tool)) {
          log(c.yellow(`     ├─ 状态:      ⚠️ 等待用户确认...`));
          const approved = await confirmCallback(toolName, toolArgs);

          if (!approved) {
            toolResult = `[ERROR] 操作被用户拒绝: ${toolName}(${JSON.stringify(toolArgs)})。请向用户解释原因并尝试其他方式。`;
            log(c.red(`     └─ 返回结果:  ❌ 用户拒绝了此操作`));
            consecutiveFailures++;
            messages.push({ role: "tool", tool_call_id: toolCall.id, content: toolResult });
            if (consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
              log(c.red(`\n  🛑 连续失败 ${consecutiveFailures} 次，强制停止重试`));
              messages.push({
                role: "system",
                content: `⚠️ 已连续失败 ${consecutiveFailures} 次。请停止尝试，向用户解释失败原因。不要再次调用工具。`,
              });
            } else {
              messages.push({
                role: "system",
                content: `⚠️ 用户拒绝了 "${toolName}" 操作。请尝试其他不需要写文件或执行命令的方式来完成用户的需求。`,
              });
            }
            continue;
          }
          log(c.yellow(`     ├─ 状态:      ✅ 用户已确认，执行中...`));
        } else {
          log(c.yellow(`     ├─ 状态:      执行中...`));
        }

        toolResult = await tool.execute(toolArgs);
        const { output, isError } = formatToolResult(toolResult);

        if (isError) {
          consecutiveFailures++;
          log(c.red(`     └─ 返回结果:  ❌ ${output}`));
          messages.push({ role: "tool", tool_call_id: toolCall.id, content: output });

          if (consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
            log(c.red(`\n  🛑 连续失败 ${consecutiveFailures} 次，强制停止重试`));
            messages.push({
              role: "system",
              content: `⚠️ 已连续失败 ${consecutiveFailures} 次。请停止尝试，向用户解释失败原因并给出可行的替代方案。不要再次调用工具。`,
            });
          } else {
            log(c.yellow(`  💡 注入反思提示（第 ${consecutiveFailures} 次失败），引导 LLM 重试...`));
            messages.push({
              role: "system",
              content: `⚠️ 工具调用失败（第 ${consecutiveFailures}/${MAX_CONSECUTIVE_FAILURES} 次）。\n失败原因: ${output}\n\n请分析失败原因，然后:\n1. 是否可以用不同的参数重试？\n2. 是否可以用其他工具实现同样的目标？\n3. 如果无法解决，向用户解释原因。`,
            });
          }
        } else {
          consecutiveFailures = 0;
          log(c.green(`     └─ 返回结果:  ${output}`));
          messages.push({ role: "tool", tool_call_id: toolCall.id, content: output });
        }
      }

      /**
       * 等待所有 delegate 子任务完成，将结果追加到 messages
       * 这里使用 Promise.all 实现并行等待
       */
      if (delegatePromises.length > 0) {
        log(c.cyan(`\n  ⏳ 等待 ${delegatePromises.length} 个子 Agent 完成...`));
        const results = await Promise.all(delegatePromises);
        for (const { id, result } of results) {
          messages.push({
            role: "tool",
            tool_call_id: id,
            content: `[SUCCESS] 子 Agent 执行完成:\n${result}`,
          });
        }
        log(c.green(`  ✅ 所有子 Agent 已完成`));
      }

      log(c.cyan("\n🔄 工具执行完毕，让 LLM 根据结果继续思考..."));
      continue;
    }

    // =========================================
    // 情况 B: LLM 给出最终文本回复（不再调用工具）
    // =========================================
    const finalAnswer = fullContent || "（LLM 未返回内容）";

    messages.push({
      role: "assistant",
      content: finalAnswer,
    });

    log(c.green("\n✅ LLM 给出了最终回复（不再调用工具）"));
    return finalAnswer;
  }

  log(c.red("\n⚠️ 达到最大循环次数限制，Agent 停止思考"));
  return "抱歉，我思考了太多轮还没有得出结论。请尝试简化问题。";
}