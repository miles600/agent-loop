/**
 * ===== Agent CLI 入口 =====
 *
 * 这是一个交互式命令行 Agent。
 * 启动后可以持续与 Agent 对话，每次对话都会完整展示 Agent 的内部思考循环。
 *
 * 特殊命令:
 *   reset  - 清空对话历史（保留系统提示词）
 *   quit   - 退出程序
 *
 * 使用方法:
 *   cd agent-cli
 *   npm start                              # 使用默认 Provider
 *   AGENT_PROVIDER=openai npm start        # 使用指定 Provider
 */

import OpenAI from "openai";
import type { ChatCompletionMessageParam } from "openai/resources/chat/completions";
import * as readline from "node:readline";
import { loadConfig, listProviders } from "./config.ts";
import { runAgentLoop, compactMessages, type ConfirmCallback } from "./agent.ts";

// 颜色输出
const c = {
  bold: (s: string) => `\x1b[1m${s}\x1b[0m`,
  dim: (s: string) => `\x1b[2m${s}\x1b[0m`,
  cyan: (s: string) => `\x1b[36m${s}\x1b[0m`,
  red: (s: string) => `\x1b[31m${s}\x1b[0m`,
  yellow: (s: string) => `\x1b[33m${s}\x1b[0m`,
  magenta: (s: string) => `\x1b[35m${s}\x1b[0m`,
};

async function main() {
  // ---- 加载 API 配置 ----
  const config = loadConfig();

  /**
   * 初始化 OpenAI 客户端。
   * 虽然是"OpenAI"客户端，但因为设置的是自定义 baseURL，
   * 所以实际上可以连接任何 OpenAI 协议兼容的服务（百度千帆、Ollama、vLLM 等）。
   */
  const client = new OpenAI({
    baseURL: config.baseURL,
    apiKey: config.apiKey,
  });

  // 列出所有可用的 Provider
  const allProviders = listProviders();

  console.log(c.bold("\n╔══════════════════════════════════════╗"));
  console.log(c.bold("║     🤖 Agent CLI - 学习版             ║"));
  console.log(c.bold("╚══════════════════════════════════════╝"));
  console.log(c.dim(`  Provider: ${c.cyan(config.provider)}`));
  console.log(c.dim(`  模型:     ${config.model}`));
  console.log(c.dim(`  API:      ${config.baseURL}`));
  console.log(c.dim(`  可用 Provider: ${allProviders.join(", ")}`));
  console.log(c.dim(`  切换方式: AGENT_PROVIDER=<name> npm start`));
  console.log(c.dim("\n  输入问题开始对话，输入 'reset' 清空记忆，'quit' 退出"));
  console.log(c.dim("  每一步都会展示 Agent 的思考-调用-观察循环\n"));

  // ---- 创建交互式命令行界面 ----
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: c.bold("\n👤 你: "),
  });

  /**
   * ===== 跨 prompt 对话记忆 =====
   *
   * messages 数组在 index.ts 中维护，而不是在 runAgentLoop 内部创建。
   * 这意味着每次用户输入新 prompt 时，LLM 都能看到之前的所有对话历史。
   *
   * 数组结构：
   *   [system, user1, assistant1, ..., userN, assistantN]
   *   如果有工具调用，中间还会穿插 tool_calls 和 tool 消息。
   */
  const messages: ChatCompletionMessageParam[] = [
    {
      role: "system",
      content: `你是一个有用的 AI 助手。你可以使用提供的工具来帮助用户解决问题。
当你需要使用工具时，直接调用对应的函数。
当你已经获得足够信息来回答用户时，用中文给出简洁、友好的回复。

## 持久记忆
项目根目录有一个 AGENTS.md 文件，是你可以读写的长期记忆。
- 需要记住重要信息时，用 file_write 写入 AGENTS.md（追加到末尾，不要覆盖）
- 每次对话开始时，用 file_read 读取 AGENTS.md 回顾之前的记忆
- AGENTS.md 最大 5000 字符，写入前请先读取确认当前长度，超出时精简旧内容`,
    },
  ];

  rl.prompt();

  // 忙等待标记：Agent 正在处理时，忽略新的用户输入，防止并发问题
  let isBusy = false;
  // 是否正在关闭中
  let isClosing = false;

  /**
   * 确认回调：当 Agent 要执行危险操作时，询问用户是否允许
   * 返回 true 表示允许执行，false 表示拒绝
   */
  const confirmCallback: ConfirmCallback = async (toolName, args) => {
    // 构建完整的执行命令预览（遍历所有参数，不写死工具名，方便扩展）
    const commandPreview = Object.entries(args)
      .map(([k, v]) => {
        const val = String(v);
        return `  ${k}: ${val.length > 200 ? val.slice(0, 200) + "..." : val}`;
      })
      .join("\n");

    console.log(c.red("\n  ╔══════════════════════════════════════╗"));
    console.log(c.red("  ║  ⚠️  危险操作确认                      ║"));
    console.log(c.red("  ╠══════════════════════════════════════╣"));
    console.log(c.red(`  ║  工具: ${toolName.padEnd(31)}║`));
    console.log(c.red(`  ║${commandPreview.padEnd(42)}║`));
    console.log(c.red("  ╚══════════════════════════════════════╝"));

    // 使用 rl.question 等待用户输入 y/n
    return new Promise((resolve) => {
      rl.question(c.yellow("\n  👆 是否允许执行? (y/n): "), (answer) => {
        resolve(answer.toLowerCase().trim() === "y");
      });
    });
  };

  // ---- 监听用户输入 ----
  rl.on("line", async (line) => {
    const input = line.trim();

    if (!input) {
      rl.prompt();
      return;
    }

    if (input.toLowerCase() === "quit" || input.toLowerCase() === "exit") {
      if (isBusy) {
        console.log(c.dim("⏳ Agent 正在思考中，请等待完成或按 Ctrl+C 强制退出...\n"));
        return;
      }
      console.log(c.dim("\n👋 再见！"));
      rl.close();
      process.exit(0);
    }

    // reset 命令：清空对话历史，只保留 system prompt
    if (input.toLowerCase() === "reset") {
      console.log(c.magenta("\n🔄 已清空对话记忆，保留系统提示词"));
      console.log(c.dim(`   清空前消息数: ${messages.length}`));
      // 只保留第一条 system 消息，删除所有后续消息
      messages.length = 1;
      console.log(c.dim(`   清空后消息数: ${messages.length}\n`));
      rl.prompt();
      return;
    }

    if (isBusy) {
      console.log(c.dim("⏳ Agent 正在思考中，请等待...\n"));
      return;
    }

    isBusy = true;
    try {
      /**
       * 把用户消息追加到共享的 messages 数组。
       * 这样 LLM 能看到之前所有的对话历史。
       */
      messages.push({ role: "user", content: input });
      // 在调用 LLM 前先裁剪过长的上下文，防止超 token 限制
      compactMessages(messages);
      await runAgentLoop(client, config, messages, confirmCallback);
    } finally {
      isBusy = false;
      if (isClosing) {
        console.log(c.dim("\n👋 Agent CLI 已关闭"));
        process.exit(0);
      }
      rl.prompt();
    }
  });

  rl.on("close", () => {
    if (isBusy) {
      console.log(c.dim("\n⏳ 等待 Agent 完成当前任务..."));
      isClosing = true;
      return;
    }
    console.log(c.dim("\n👋 Agent CLI 已关闭"));
    process.exit(0);
  });
}

main().catch((err) => {
  console.error("启动失败:", err.message);
  process.exit(1);
});