import OpenAI from "openai";
import type { ChatCompletionMessageParam } from "openai/resources/chat/completions";
import type { AgentConfig } from "./config.ts";
import { allTools, findTool, isDangerous } from "./tools.ts";

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

/**
 * ===== Agent 循环（流式输出 + 共享消息历史） =====
 *
 * 这是 Agent 的核心：一个"思考-行动-观察"的循环。
 *
 * 与之前版本的关键区别：
 *   messages 数组由调用方传入并维护，不再在函数内部创建。
 *   这意味着多次调用 runAgentLoop 可以共享同一个 messages 数组，
 *   实现跨 prompt 的对话记忆。
 *
 * 流程：
 *   1. 使用传入的 messages 数组（已包含 system + 历史 + 新 user 消息）调用 LLM
 *   2. 实时流式接收 LLM 的响应，逐 token 打印到终端
 *   3. 流结束后判断：
 *      a) 文本回复 → 把 assistant 消息追加到 messages，退出循环
 *      b) 工具调用 → 进入步骤 4
 *   4. 逐个检查工具是否需要确认：
 *      - safe 工具 → 直接执行
 *      - dangerous 工具 → 调用 confirmCallback 询问用户
 *   5. 执行工具，把 assistant(tool_calls) 和 tool(result) 追加到 messages
 *   6. 回到步骤 1，让 LLM 根据新信息继续思考
 *
 * 这个循环会一直运行，直到 LLM 决定不再调用工具（给出最终回复）。
 * 设置了 maxTurns 防止无限循环。
 */
export async function runAgentLoop(
  client: OpenAI,
  config: AgentConfig,
  messages: ChatCompletionMessageParam[], // 共享的对话历史，由调用方维护
  confirmCallback: ConfirmCallback,
): Promise<string> {
  // 安全阀：最多循环 10 轮，防止无限调用工具
  const MAX_TURNS = 30;

  for (let turn = 0; turn < MAX_TURNS; turn++) {
    console.log(c.dim(`\n--- Agent 循环 第 ${turn + 1} 轮 ---`));

    // =========================================
    // 步骤 1: 流式调用 LLM
    // =========================================
    console.log(c.cyan("📡 发送请求到 LLM（流式模式）..."));
    console.log(c.dim(`   模型: ${config.model}`));
    console.log(c.dim(`   当前对话消息数: ${messages.length}`));

    /**
     * stream: true 让 LLM 不等待全部生成完毕再返回，而是边生成边发送。
     * 返回的是一个异步可迭代对象（AsyncIterable），每个 chunk 是一个小片段。
     */
    const stream = await client.chat.completions.create({
      model: config.model,
      messages,
      tools: allTools.map((t) => ({
        type: "function" as const,
        function: t.definition,
      })),
      tool_choice: "auto",
      stream: true, // 启用流式输出
    });

    // ---- 累积器：从流式 chunk 中拼装完整的响应 ----
    let fullContent = ""; // 累积文本内容
    // 工具调用需要在 Map 中累积，因为可能有多个 tool_call 同时到达
    const toolCallAcc = new Map<number, AccumulatedToolCall>();

    process.stdout.write(c.blue("🤖 ")); // 打印 Agent 回复前缀

    // 遍历流中的每一个 chunk
    for await (const chunk of stream) {
      const delta = chunk.choices?.[0]?.delta;
      if (!delta) continue;

      /**
       * 文本内容增量（delta.content）
       * 每个 chunk 可能包含几个字的文本，直接追加到终端即可实现打字机效果
       */
      if (delta.content) {
        fullContent += delta.content;
        process.stdout.write(delta.content); // 逐 token 输出到终端
      }

      /**
       * 工具调用增量（delta.tool_calls）
       * 工具调用的数据是分多个 chunk 到达的：
       *   - 第一个 chunk 携带 id 和 function.name
       *   - 后续 chunk 携带 function.arguments 的片段（JSON 字符串的一部分）
       * 需要把这些片段拼成完整的工具调用
       */
      if (delta.tool_calls) {
        for (const tc of delta.tool_calls) {
          const idx = tc.index;
          // 如果这个索引的工具调用还不存在，初始化一个空的累积器
          if (!toolCallAcc.has(idx)) {
            toolCallAcc.set(idx, { id: "", name: "", arguments: "" });
          }
          const acc = toolCallAcc.get(idx)!;
          if (tc.id) acc.id = tc.id; // id 只在第一个 chunk 出现
          if (tc.function?.name) acc.name += tc.function.name; // name 只在第一个 chunk 出现
          if (tc.function?.arguments) acc.arguments += tc.function.arguments; // arguments 可能跨多个 chunk
        }
      }
    }

    // 流结束，输出换行
    if (fullContent) {
      process.stdout.write("\n");
    }

    // ---- 判断流结束后的响应类型 ----

    /**
     * 情况 A: LLM 决定调用工具
     * 工具调用的累积器中有数据，说明 LLM 想要调用工具
     */
    if (toolCallAcc.size > 0) {
      console.log(
        c.yellow(`\n🔧 LLM 决定调用 ${toolCallAcc.size} 个工具:`),
      );

      // 将累积的工具调用拼装成 OpenAI API 兼容的格式
      const toolCalls = Array.from(toolCallAcc.entries())
        .sort(([a], [b]) => a - b) // 按 index 排序，保证顺序
        .map(([_, tc]) => ({
          id: tc.id,
          type: "function" as const,
          function: {
            name: tc.name,
            arguments: tc.arguments, // 累积完成的 JSON 字符串
          },
        }));

      // 先记录 LLM 的工具调用请求到对话历史
      messages.push({
        role: "assistant",
        content: fullContent || null,
        tool_calls: toolCalls,
      });

      /**
       * 步骤 2: 逐个执行每个工具
       *
       * OpenAI 允许一次性返回多个 tool_calls，我们需要逐个处理。
       * 每个工具执行后，将结果以 role: "tool" 的消息格式追加到对话历史。
       */
      for (const toolCall of toolCalls) {
        const toolName = toolCall.function.name;
        // LLM 返回的参数是 JSON 字符串，需要解析
        const toolArgs = JSON.parse(toolCall.function.arguments);
        const tool = findTool(toolName);

        // 打印详细的工具调用信息
        console.log(c.yellow(`\n  📞 工具调用链:  ${toolName}`));
        console.log(c.yellow(`     ├─ ID:        ${toolCall.id}`));
        console.log(c.yellow(`     ├─ 参数:      ${JSON.stringify(toolArgs)}`));

        let toolResult: string;

        if (!tool) {
          toolResult = `错误: 未找到工具 "${toolName}"`;
          console.log(c.red(`     └─ 返回结果:  ${toolResult}`));
        } else {
          // 打印风险等级
          const riskLabel = isDangerous(tool) ? "🔴 危险" : "🟢 安全";
          console.log(c.yellow(`     ├─ 风险等级:  ${riskLabel}`));

          /**
           * 步骤 3: 权限检查
           * 危险工具（file_write、run_bash）需要用户确认后才能执行
           */
          if (isDangerous(tool)) {
            console.log(c.yellow(`     ├─ 状态:      ⚠️ 等待用户确认...`));
            const approved = await confirmCallback(toolName, toolArgs);

            if (!approved) {
              // 用户拒绝执行，告诉 LLM 操作被取消
              toolResult = `操作被用户拒绝: ${toolName}(${JSON.stringify(toolArgs)})。请向用户解释原因并尝试其他方式。`;
              console.log(c.red(`     └─ 返回结果:  ❌ 用户拒绝了此操作`));
              messages.push({
                role: "tool",
                tool_call_id: toolCall.id,
                content: toolResult,
              });
              continue;
            }
            console.log(c.yellow(`     ├─ 状态:      ✅ 用户已确认，执行中...`));
          } else {
            console.log(c.yellow(`     ├─ 状态:      执行中...`));
          }

          // 步骤 4: 执行工具
          toolResult = await tool.execute(toolArgs);
          console.log(c.green(`     └─ 返回结果:  ${toolResult}`));
        }

        // 步骤 5: 把工具执行结果追加到对话历史
        messages.push({
          role: "tool",
          tool_call_id: toolCall.id,
          content: toolResult,
        });
      }

      // 回到循环开头，让 LLM 根据工具结果重新思考
      console.log(c.cyan("\n🔄 工具执行完毕，让 LLM 根据结果继续思考..."));
      continue;
    }

    // =========================================
    // 情况 B: LLM 给出最终文本回复（不再调用工具）
    // 文本已经在流式接收时实时打印到终端了
    // =========================================
    const finalAnswer = fullContent || "（LLM 未返回内容）";

    // 把 LLM 的最终回复也追加到对话历史，下次对话 LLM 能看到
    messages.push({
      role: "assistant",
      content: finalAnswer,
    });

    console.log(c.green("\n✅ LLM 给出了最终回复（不再调用工具）"));
    return finalAnswer;
  }

  // 如果超过了最大循环次数仍未得到最终答案
  console.log(c.red("\n⚠️ 达到最大循环次数限制，Agent 停止思考"));
  return "抱歉，我思考了太多轮还没有得出结论。请尝试简化问题。";
}