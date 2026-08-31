import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    exclude: ['node_modules/**', 'dist/**', '.sessions/**'],
    setupFiles: ['src/test-setup.ts'],
  },
});
