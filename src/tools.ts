/**
 * ===== 工具定义与执行 =====
 *
 * 这里的每一个工具都实现了 OpenAI 的 Function Calling 协议：
 *   1. definition  - 告诉 LLM 这个工具叫什么、有什么参数、做什么用
 *   2. execute     - Agent 循环中实际执行这个工具的代码
 *
 * 你可以按这个格式随意添加新工具，Agent 会自动发现并使用它们。
 */

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
 * 完整的工具描述：定义 + 执行函数
 */
export interface Tool {
  definition: ToolDefinition;
  execute: ToolExecutor;
}

// ============================================================
// 工具 1: 模拟天气查询
// ============================================================
const getWeatherTool: Tool = {
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
    // 模拟不同城市的天气数据（实际应用中这里会调用真实天气 API）
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
      .replace(/\^/g, "**") // 将 ^ 转为 JS 的 ** 幂运算符
      .replace(/[^0-9+\-*/().%\s]/g, ""); // 安全过滤，只允许数学字符
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
// 所有已注册的工具列表
// 添加新工具只需在这里追加即可，Agent 会自动感知
// ============================================================
export const allTools: Tool[] = [getWeatherTool, calculatorTool, getCurrentTimeTool];

/**
 * 根据工具名查找工具
 */
export function findTool(name: string): Tool | undefined {
  return allTools.find((t) => t.definition.name === name);
}