# agent-loop

一个手写 Agent Loop 的学习项目，用 TypeScript 从零实现 Agent 的 **思考 → 调用工具 → 观察结果 → 再思考** 循环。

不使用任何 Agent 框架，只依赖 OpenAI 协议兼容的 API 和 3 个示例工具，<150 行核心代码展示 Agent 的本质。

## 运行

```bash
npm install
npm start
```

启动后输入问题，终端会实时展示 Agent 每一步的决策过程。

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
src/
├── config.ts   # 读取 YAML 配置，支持多 Provider
├── tools.ts    # 工具定义 + 执行器（天气、计算器、时间）
├── agent.ts    # Agent 核心循环（思考-调用-观察）
└── index.ts    # 交互式 CLI 入口
```

## 添加新工具

在 `src/tools.ts` 中追加一个 `Tool` 对象即可：

```ts
const myTool: Tool = {
  definition: {
    name: "my_tool",
    description: "工具描述",
    parameters: {
      type: "object",
      properties: { ... },
      required: [...],
    },
  },
  execute: async (args) => { ... },
};

export const allTools: Tool[] = [..., myTool];
```

## License

MIT