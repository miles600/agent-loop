# agent-loop

一个手写 Agent Loop 的学习项目，用 TypeScript 从零实现 Agent 的完整核心能力。

**不使用任何 Agent 框架**，只依赖 OpenAI 协议兼容的 API，逐步展示 Agent 的每一个原理。

## 已实现功能

| 功能 | 说明 |
|------|------|
| **工具调用** | 7 个工具（天气、计算器、时间、文件读写、Shell、网页抓取、委派子 Agent） |
| **权限控制** | 危险工具（file_write、run_bash）执行前需用户确认，安全工具直接执行 |
| **流式输出** | 基于 SSE，LLM 回复逐 token 实时显示 |
| **跨轮记忆** | 同一会话中多次 prompt 共享对话历史，`reset` 命令可清空 |
| **持久记忆** | `AGENTS.md` 文件可被 Agent 读写，跨会话保存信息 |
| **上下文裁剪** | 对话历史超限时自动裁剪旧消息，保留 system prompt + 最近消息 |
| **反思/纠错** | 工具调用失败后 LLM 自动分析原因、换策略重试，连续 3 次失败强制停止 |
| **多 Agent 协作** | `delegate` 工具可委派子 Agent 并行执行任务，独立上下文，不污染主 Agent 记忆 |

## 运行

```bash
cd agent-cli
npm install
npm start
```

启动后输入问题，终端会实时展示 Agent 每一步的决策过程。

### 特殊命令

| 命令 | 作用 |
|------|------|
| `reset` | 清空对话历史（保留系统提示词） |
| `quit` / `exit` | 退出程序 |

## 配置

在项目**上级目录**创建 `api_keys.yaml`（不上传 git）：

```yaml
default: baidu

providers:
  baidu:
    url: 'https://qianfan.baidubce.com/v2/tokenplan/personal'
    api_key: 'your-api-key'
    model: 'deepseek-v4-pro'
  ollama:
    url: 'http://localhost:11434/v1'
    api_key: 'ollama'
    model: 'qwen2.5:7b'
```

切换 Provider：`AGENT_PROVIDER=ollama npm start`

## 项目结构

```
agent-cli/
├── AGENTS.md      # Agent 持久记忆文件（可被 Agent 读写）
├── src/
│   ├── config.ts  # 读取 YAML 配置，支持多 Provider
│   ├── tools.ts   # 工具定义 + 执行器 + 权限等级
│   ├── agent.ts   # Agent 核心循环 + 上下文管理 + 反思纠错 + 多 Agent
│   └── index.ts   # 交互式 CLI 入口 + 跨轮记忆
├── package.json
└── tsconfig.json
```

## 核心原理

Agent 的本质是一个 `for` 循环：

```
for (turn = 0; turn < MAX_TURNS; turn++) {
  1. 把 messages 发给 LLM
  2. LLM 返回 text   → 退出循环，输出最终回复
  3. LLM 返回 tool_calls → 逐个执行工具，结果追加到 messages
  4. 回到第 1 步
}
```

在 `src/agent.ts` 的 `runAgentLoop` 函数中实现，注释详细。

## 添加新工具

在 `src/tools.ts` 中追加一个 `Tool` 对象即可：

```ts
const myTool: Tool = {
  riskLevel: "safe",  // "safe" 直接执行，"dangerous" 需确认
  definition: {
    name: "my_tool",
    description: "工具描述",
    parameters: {
      type: "object",
      properties: { /* ... */ },
      required: [/* ... */],
    },
  },
  execute: async (args) => {
    // 工具执行逻辑
    return "执行结果";
  },
};

export const allTools: Tool[] = [..., myTool];
```

## License

MIT