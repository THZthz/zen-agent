/**
 * Global vitest setup: configure default user-defined providers so tests
 * that create sessions (and therefore resolve the default provider) work
 * without each one declaring providers. Individual tests override
 * ZEN_AGENT_PROVIDERS as needed and restore env afterwards.
 */
if (!process.env.ZEN_AGENT_PROVIDERS) {
  process.env.ZEN_AGENT_PROVIDERS = JSON.stringify([
    {
      id: 'deepseek',
      name: 'DeepSeek',
      baseUrl: 'https://api.deepseek.com',
      apiKeyEnv: 'DEEPSEEK_API_KEY',
      defaultModel: 'deepseek-v4-flash',
      models: [
        { id: 'deepseek-v4-flash', name: 'DeepSeek V4 Flash', contextLength: 1_000_000 },
        { id: 'deepseek-v4-pro', name: 'DeepSeek V4 Pro', contextLength: 1_000_000 },
      ],
    },
    {
      id: 'openrouter',
      name: 'OpenRouter',
      baseUrl: 'https://openrouter.ai/api/v1',
      apiKeyEnv: 'OPENROUTER_API_KEY',
      defaultModel: 'openrouter/free',
      models: [{ id: 'openrouter/free', name: 'OpenRouter Free', contextLength: 128_000 }],
    },
  ]);
}
