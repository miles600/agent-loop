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
 * ===== Agent 循环 =====
 *
 * 这是 Agent 的核心：一个"思考-行动-观察"的循环。
 *
 * 流程：
 *   1. 把对话历史发给 LLM
 *   2. LLM 返回两种可能：
 *      a) 一段文本回复 → 任务完成，退出循环
 *      b) 一个或多个工具调用请求 → 进入步骤 3
 *   3. 逐个检查工具是否需要确认：
 *      - safe 工具 → 直接执行
 *      - dangerous 工具 → 调用 confirmCallback 询问用户
 *   4. 执行工具，得到结果
 *   5. 把工具执行结果追加到对话历史中
 *   6. 回到步骤 1，让 LLM 根据新信息继续思考
 *
 * 这个循环会一直运行，直到 LLM 决定不再调用工具（给出最终回复）。
 * 设置了 maxTurns 防止无限循环。
 */
export async function runAgentLoop(
  client: OpenAI,
  config: AgentConfig,
  userMessage: string,
  confirmCallback: ConfirmCallback,
): Promise<string> {
  // ---- 系统提示词：告诉 LLM 它的角色和行为规则 ----
  const systemPrompt = `你是一个有用的 AI 助手。你可以使用提供的工具来帮助用户解决问题。
当你需要使用工具时，直接调用对应的函数。
当你已经获得足够信息来回答用户时，用中文给出简洁、友好的回复。`;

  /**
   * 对话历史数组。
   * 这是 Agent 的"记忆"——LLM 每次调用都能看到之前的所有对话，
   * 包括用户说的话、LLM 的回复、以及工具调用的结果。
   *
   * 格式遵循 OpenAI Chat Completions API 的 message 格式：
   *   { role: "system" | "user" | "assistant" | "tool", content: "..." }
   */
  const messages: ChatCompletionMessageParam[] = [
    { role: "system", content: systemPrompt },
    { role: "user", content: userMessage },
  ];

  // 安全阀：最多循环 10 轮，防止无限调用工具
  const MAX_TURNS = 10;

  for (let turn = 0; turn < MAX_TURNS; turn++) {
    console.log(c.dim(`\n--- Agent 循环 第 ${turn + 1} 轮 ---`));

    // =========================================
    // 步骤 1: 调用 LLM，传入当前对话历史
    // =========================================
    console.log(c.cyan("📡 发送请求到 LLM..."));
    console.log(c.dim(`   模型: ${config.model}`));
    console.log(c.dim(`   当前对话消息数: ${messages.length}`));

    const response = await client.chat.completions.create({
      model: config.model,
      messages,
      // 把工具定义告诉 LLM，让它知道可以调用哪些工具
      tools: allTools.map((t) => ({
        type: "function" as const,
        function: t.definition,
      })),
      // 让 LLM 自行决定是否调用工具（auto 模式）
      tool_choice: "auto",
    });

    const choice = response.choices[0];
    if (!choice) {
      console.log(c.red("❌ LLM 返回为空"));
      return "抱歉，请求失败。";
    }

    const assistantMessage = choice.message;

    // =========================================
    // 情况 A: LLM 决定调用工具
    // =========================================
    if (assistantMessage.tool_calls && assistantMessage.tool_calls.length > 0) {
      console.log(
        c.yellow(`🔧 LLM 决定调用 ${assistantMessage.tool_calls.length} 个工具:`),
      );

      // 先记录 LLM 的工具调用请求到对话历史
      messages.push({
        role: "assistant",
        content: assistantMessage.content,
        tool_calls: assistantMessage.tool_calls,
      });

      /**
       * 步骤 2: 逐个执行每个工具
       *
       * OpenAI 允许一次性返回多个 tool_calls，我们需要逐个处理。
       * 每个工具执行后，将结果以 role: "tool" 的消息格式追加到对话历史。
       */
      for (const toolCall of assistantMessage.tool_calls) {
        const toolName = toolCall.function.name;
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
    // =========================================
    const finalAnswer = assistantMessage.content ?? "（LLM 未返回内容）";
    console.log(c.green("\n✅ LLM 给出了最终回复（不再调用工具）"));
    console.log(c.blue(`\n🤖 Agent 回复:\n${finalAnswer}`));
    return finalAnswer;
  }

  // 如果超过了最大循环次数仍未得到最终答案
  console.log(c.red("\n⚠️ 达到最大循环次数限制，Agent 停止思考"));
  return "抱歉，我思考了太多轮还没有得出结论。请尝试简化问题。";
}