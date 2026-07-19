/**
 * ===== 工具定义与执行 =====
 *
 * 这里的每一个工具都实现了 OpenAI 的 Function Calling 协议：
 *   1. definition  - 告诉 LLM 这个工具叫什么、有什么参数、做什么用
 *   2. execute     - Agent 循环中实际执行这个工具的代码
 *   3. riskLevel   - 权限等级，决定执行前是否需要用户确认
 *
 * 你可以按这个格式随意添加新工具，Agent 会自动发现并使用它们。
 */

import { readFileSync, writeFileSync, existsSync } from "node:fs";

/**
 * 工具的定义接口 —— 对应 OpenAI Function Calling 的 function 字段
 */
export interface ToolDefinition {
  name: string;
  description: string;
  parameters: {
    type: "object";
    properties: Record<string, { type: string; description: string }>;
    required: string[];
  };
}

/**
 * 工具执行器 —— 接收参数，返回一段文本结果
 */
export type ToolExecutor = (args: Record<string, unknown>) => Promise<string> | string;

/**
 * 权限等级
 * - "safe":      安全操作，直接执行（查询、计算等）
 * - "dangerous": 危险操作，执行前必须询问用户确认（写文件、执行命令等）
 */
export type RiskLevel = "safe" | "dangerous";

/**
 * 完整的工具描述：定义 + 执行函数 + 权限等级
 */
export interface Tool {
  definition: ToolDefinition;
  execute: ToolExecutor;
  riskLevel: RiskLevel;
}

// ============================================================
// 工具 1: 模拟天气查询
// ============================================================
const getWeatherTool: Tool = {
  riskLevel: "safe",
  definition: {
    name: "get_weather",
    description: "查询指定城市的天气信息，返回温度、天气状况、湿度等",
    parameters: {
      type: "object",
      properties: {
        city: {
          type: "string",
          description: "城市名称，例如 '北京'、'上海'、'深圳'",
        },
      },
      required: ["city"],
    },
  },
  execute: async (args) => {
    const city = args["city"] as string;
    const weatherDB: Record<string, { temp: number; weather: string; humidity: number }> = {
      "北京": { temp: 32, weather: "晴", humidity: 45 },
      "上海": { temp: 29, weather: "多云转小雨", humidity: 70 },
      "深圳": { temp: 33, weather: "雷阵雨", humidity: 85 },
      "东京": { temp: 27, weather: "阴", humidity: 60 },
      "纽约": { temp: 22, weather: "晴", humidity: 40 },
    };
    const info = weatherDB[city] ?? { temp: 25, weather: "未知", humidity: 50 };
    return `${city}当前天气：${info.weather}，温度 ${info.temp}°C，湿度 ${info.humidity}%`;
  },
};

// ============================================================
// 工具 2: 计算器
// ============================================================
const calculatorTool: Tool = {
  riskLevel: "safe",
  definition: {
    name: "calculate",
    description: "执行数学计算。支持的运算：加法(+)、减法(-)、乘法(*)、除法(/)、幂(^)",
    parameters: {
      type: "object",
      properties: {
        expression: {
          type: "string",
          description: "数学表达式，例如 '3 + 5 * 2'、'(10 - 3) * 4'、'2 ^ 8'",
        },
      },
      required: ["expression"],
    },
  },
  execute: async (args) => {
    const expr = (args["expression"] as string)
      .replace(/\^/g, "**")
      .replace(/[^0-9+\-*/().%\s]/g, "");
    try {
      // eslint-disable-next-line no-eval
      const result = eval(expr);
      return `${args["expression"]} = ${result}`;
    } catch {
      return `无法计算表达式: ${args["expression"]}`;
    }
  },
};

// ============================================================
// 工具 3: 获取当前时间
// ============================================================
const getCurrentTimeTool: Tool = {
  riskLevel: "safe",
  definition: {
    name: "get_current_time",
    description: "获取当前日期和时间，可以指定时区",
    parameters: {
      type: "object",
      properties: {
        timezone: {
          type: "string",
          description: "时区，例如 'Asia/Shanghai'、'America/New_York'，默认为 Asia/Shanghai",
        },
      },
      required: [],
    },
  },
  execute: async (args) => {
    const timezone = (args["timezone"] as string) || "Asia/Shanghai";
    const now = new Date();
    const formatted = now.toLocaleString("zh-CN", { timeZone: timezone });
    return `当前时间 (${timezone}): ${formatted}`;
  },
};

// ============================================================
// 工具 4: 读取文件
// ============================================================
const fileReadTool: Tool = {
  riskLevel: "safe",
  definition: {
    name: "file_read",
    description:
      "读取指定路径的文件内容。支持文本文件和代码文件。返回文件内容（最多 5000 字符）。",
    parameters: {
      type: "object",
      properties: {
        path: {
          type: "string",
          description: "文件路径，可以是绝对路径或相对于当前工作目录的路径",
        },
      },
      required: ["path"],
    },
  },
  execute: async (args) => {
    const path = args["path"] as string;
    try {
      if (!existsSync(path)) {
        return `错误: 文件不存在 - "${path}"`;
      }
      const content = readFileSync(path, "utf-8");
      if (content.length > 5000) {
        return content.slice(0, 5000) + "\n\n... (文件内容已截断，共 " + content.length + " 字符)";
      }
      return `文件 "${path}" 的内容:\n\n${content}`;
    } catch (err) {
      return `读取文件失败: ${(err as Error).message}`;
    }
  },
};

// ============================================================
// 工具 5: 写入文件（需要确认）
// ============================================================
const fileWriteTool: Tool = {
  riskLevel: "dangerous",
  definition: {
    name: "file_write",
    description: "将内容写入指定路径的文件。如果文件不存在则创建，如果存在则覆盖。",
    parameters: {
      type: "object",
      properties: {
        path: {
          type: "string",
          description: "文件路径，可以是绝对路径或相对于当前工作目录的路径",
        },
        content: {
          type: "string",
          description: "要写入的文件内容",
        },
      },
      required: ["path", "content"],
    },
  },
  execute: async (args) => {
    const path = args["path"] as string;
    const content = args["content"] as string;
    try {
      writeFileSync(path, content, "utf-8");
      return `文件写入成功: "${path}" (${content.length} 字符)`;
    } catch (err) {
      return `写入文件失败: ${(err as Error).message}`;
    }
  },
};

// ============================================================
// 工具 6: 执行 Shell 命令（需要确认）
// ============================================================
const runBashTool: Tool = {
  riskLevel: "dangerous",
  definition: {
    name: "run_bash",
    description: "在终端中执行一条 shell 命令，返回命令的输出结果。",
    parameters: {
      type: "object",
      properties: {
        command: {
          type: "string",
          description: "要执行的 shell 命令，例如 'ls -la'、'cat README.md'、'echo hello'",
        },
      },
      required: ["command"],
    },
  },
  execute: async (args) => {
    const command = args["command"] as string;
    try {
      const { execSync } = await import("node:child_process");
      const output = execSync(command, {
        encoding: "utf-8",
        timeout: 10000,             // 10 秒超时
        maxBuffer: 1024 * 1024,     // 1MB 输出上限
      });
      return `命令执行成功:\n${output || "(无输出)"}`;
    } catch (err) {
      return `命令执行失败: ${(err as Error).message}`;
    }
  },
};

// ============================================================
// 工具 7: 网页抓取
// ============================================================
const webFetchTool: Tool = {
  riskLevel: "safe",
  definition: {
    name: "web_fetch",
    description: "抓取指定 URL 的网页内容，返回纯文本格式（最多 5000 字符）。",
    parameters: {
      type: "object",
      properties: {
        url: {
          type: "string",
          description: "要抓取的网页 URL，必须以 http:// 或 https:// 开头",
        },
      },
      required: ["url"],
    },
  },
  execute: async (args) => {
    const url = args["url"] as string;
    if (!url.startsWith("http://") && !url.startsWith("https://")) {
      return "错误: URL 必须以 http:// 或 https:// 开头";
    }
    try {
      const response = await fetch(url, {
        headers: { "User-Agent": "agent-loop/1.0" },
        signal: AbortSignal.timeout(10000), // 10 秒超时
      });
      if (!response.ok) {
        return `抓取失败: HTTP ${response.status} ${response.statusText}`;
      }
      const html = await response.text();
      // 简单去除 HTML 标签，提取纯文本
      const text = html
        .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, "")
        .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, "")
        .replace(/<[^>]+>/g, " ")
        .replace(/&nbsp;/g, " ")
        .replace(/&amp;/g, "&")
        .replace(/&lt;/g, "<")
        .replace(/&gt;/g, ">")
        .replace(/\s+/g, " ")
        .trim();
      if (text.length > 5000) {
        return text.slice(0, 5000) + "\n\n... (网页内容已截断，共 " + text.length + " 字符)";
      }
      return `网页 "${url}" 的内容:\n\n${text}`;
    } catch (err) {
      return `抓取失败: ${(err as Error).message}`;
    }
  },
};

// ============================================================
// 工具 8: 委派子 Agent
// 注意：这个工具的 execute 不会被正常调用，
// 由 agent.ts 中的 runAgentLoop 特殊处理 —— 创建独立上下文运行子 Agent 循环
// ============================================================
const delegateTool: Tool = {
  riskLevel: "safe",
  definition: {
    name: "delegate",
    description:
      "委派一个子 Agent 独立执行复杂任务。子 Agent 拥有独立的对话上下文，不会污染主 Agent 的记忆。可以同时委派多个子 Agent 并行执行。子 Agent 完成后返回结果摘要。",
    parameters: {
      type: "object",
      properties: {
        task: {
          type: "string",
          description:
            "子 Agent 的系统提示词，描述它的角色、目标和约束。例如 '你是一个代码审查专家，请分析 README.md 的代码质量并给出改进建议'",
        },
        context: {
          type: "string",
          description: "给子 Agent 的补充信息和具体指令，例如 '重点关注错误处理和性能优化'",
        },
        tools: {
          type: "string",
          description:
            "子 Agent 可以使用的工具列表，逗号分隔。例如 'file_read,calculate,web_fetch'。不填则使用所有安全工具（不含 delegate）。",
        },
      },
      required: ["task"],
    },
  },
  execute: async () => {
    // delegate 工具由 agent.ts 特殊处理，这里的 execute 不会被调用
    return "[INTERNAL] delegate 工具不应通过正常路径执行";
  },
};

// ============================================================
// 所有已注册的工具列表
// 添加新工具只需在这里追加即可，Agent 会自动感知
// ============================================================
export const allTools: Tool[] = [
  getWeatherTool,
  calculatorTool,
  getCurrentTimeTool,
  fileReadTool,
  fileWriteTool,
  runBashTool,
  webFetchTool,
  delegateTool,
];

/**
 * 获取除去 delegate 的工具列表（子 Agent 不能用 delegate 递归委派）
 */
export function getToolsWithoutDelegate(): Tool[] {
  return allTools.filter((t) => t.definition.name !== "delegate");
}

/**
 * 根据工具名查找工具
 */
export function findTool(name: string): Tool | undefined {
  return allTools.find((t) => t.definition.name === name);
}

/**
 * 判断工具是否需要用户确认才能执行
 */
export function isDangerous(tool: Tool): boolean {
  return tool.riskLevel === "dangerous";
}