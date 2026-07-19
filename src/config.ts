import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { load as loadYaml } from "js-yaml";

/**
 * ===== Agent 多 Provider 配置 =====
 *
 * 从项目根目录的 api_keys.yaml 中读取所有 Provider 配置。
 * 选择优先级: 环境变量 AGENT_PROVIDER > YAML 中的 default 字段
 *
 * YAML 格式示例:
 *   default: baidu
 *   providers:
 *     baidu:
 *       url: 'https://...'
 *       api_key: 'xxx'
 *       model: 'deepseek-v4-pro'
 *     openai:
 *       url: 'https://api.openai.com/v1'
 *       api_key: 'sk-xxx'
 *       model: 'gpt-4o'
 */

// api_keys.yaml 位于 agent-cli 的父目录
const __dirname = dirname(fileURLToPath(import.meta.url));
const CONFIG_FILE = resolve(__dirname, "../../api_keys.yaml");

/** 单个 Provider 的配置 */
export interface AgentConfig {
  /** Provider 名称（用于显示） */
  provider: string;
  /** OpenAI 协议兼容的 API 地址 */
  baseURL: string;
  /** API 密钥 */
  apiKey: string;
  /** 模型名称 */
  model: string;
}

/** YAML 文件的顶层结构 */
interface YamlConfig {
  default?: string;
  providers: Record<string, { url: string; api_key: string; model: string }>;
}

/**
 * 从 api_keys.yaml 加载配置，按 AGENT_PROVIDER 环境变量选择 Provider
 */
export function loadConfig(): AgentConfig {
  const raw = readFileSync(CONFIG_FILE, "utf-8");
  const yaml = loadYaml(raw) as YamlConfig;

  if (!yaml.providers || Object.keys(yaml.providers).length === 0) {
    throw new Error("api_keys.yaml 中没有配置任何 Provider，请在 providers 下添加至少一个");
  }

  // 选择优先级: 环境变量 AGENT_PROVIDER > YAML 的 default 字段
  const targetProvider = process.env["AGENT_PROVIDER"] ?? yaml.default;
  if (!targetProvider) {
    const available = Object.keys(yaml.providers).join(", ");
    throw new Error(
      "未指定 Provider。请设置环境变量 AGENT_PROVIDER 或在 api_keys.yaml 中配置 default 字段\n" +
        `可用的 Provider: ${available}`,
    );
  }
  const provider = yaml.providers[targetProvider];

  if (!provider) {
    const available = Object.keys(yaml.providers).join(", ");
    throw new Error(
      `未找到 Provider "${targetProvider}"，可用的 Provider: ${available}\n` +
        `可以通过 AGENT_PROVIDER 环境变量切换，例如: AGENT_PROVIDER=openai npm start`,
    );
  }

  return {
    provider: targetProvider,
    baseURL: provider.url,
    apiKey: provider.api_key,
    model: provider.model,
  };
}

/**
 * 列出所有可用的 Provider 名称（用于启动时显示）
 */
export function listProviders(): string[] {
  const raw = readFileSync(CONFIG_FILE, "utf-8");
  const yaml = loadYaml(raw) as YamlConfig;
  return Object.keys(yaml.providers);
}