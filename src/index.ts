/**
 * ===== Agent CLI 入口 =====
 *
 * 这是一个交互式命令行 Agent。
 * 启动后可以持续与 Agent 对话，每次对话都会完整展示 Agent 的内部思考循环。
 *
 * 使用方法:
 *   cd agent-cli
 *   npm start                              # 使用默认 Provider
 *   AGENT_PROVIDER=openai npm start        # 使用指定 Provider
 */

import OpenAI from "openai";
import * as readline from "node:readline";
import { loadConfig, listProviders } from "./config.ts";
import { runAgentLoop, type ConfirmCallback } from "./agent.ts";

// 颜色输出
const c = {
  bold: (s: string) => `\x1b[1m${s}\x1b[0m`,
  dim: (s: string) => `\x1b[2m${s}\x1b[0m`,
  cyan: (s: string) => `\x1b[36m${s}\x1b[0m`,
  red: (s: string) => `\x1b[31m${s}\x1b[0m`,
  yellow: (s: string) => `\x1b[33m${s}\x1b[0m`,
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
  console.log(c.dim("\n  输入问题开始对话，输入 'quit' 或 Ctrl+C 退出"));
  console.log(c.dim("  每一步都会展示 Agent 的思考-调用-观察循环\n"));

  // ---- 创建交互式命令行界面 ----
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
    prompt: c.bold("\n👤 你: "),
  });

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

    if (isBusy) {
      console.log(c.dim("⏳ Agent 正在思考中，请等待...\n"));
      return;
    }

    isBusy = true;
    try {
      await runAgentLoop(client, config, input, confirmCallback);
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