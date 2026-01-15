#!/usr/bin/env bun

import * as readline from "readline";

// ANSI color codes
const colors = {
  reset: "\x1b[0m",
  cyan: "\x1b[36m",
  green: "\x1b[32m",
  yellow: "\x1b[33m",
  red: "\x1b[31m",
  dim: "\x1b[2m",
};

// Get API key from environment
const API_KEY = process.env.FIREWORKS_API_KEY;
if (!API_KEY || API_KEY === "your-api-key-here") {
  console.error(`${colors.red}Error: FIREWORKS_API_KEY not set in .env file${colors.reset}`);
  console.error(`${colors.dim}Please set FIREWORKS_API_KEY=your-actual-key in .env${colors.reset}`);
  process.exit(1);
}

// TypeScript assertion: API_KEY is guaranteed to be a string after the check above
const FIREWORKS_API_KEY: string = API_KEY;

const API_URL = "https://api.fireworks.ai/inference/v1/chat/completions";

interface Message {
  role: "user" | "assistant" | "system";
  content: string;
}

class FireworksClient {
  private apiKey: string;
  private apiUrl: string;

  constructor(apiKey: string) {
    this.apiKey = apiKey;
    this.apiUrl = API_URL;
  }

  async *streamChat(messages: Message[]): AsyncGenerator<string, void, unknown> {
    const response = await fetch(this.apiUrl, {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        model: "accounts/fireworks/models/gpt-oss-20b",
        max_tokens: 16384,
        top_p: 1,
        top_k: 40,
        presence_penalty: 0,
        frequency_penalty: 0,
        temperature: 0.6,
        messages,
        stream: true,
      }),
    });

    if (!response.ok) {
      const error = await response.text();
      throw new Error(`API Error: ${response.status} ${response.statusText}\n${error}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error("No response body");
    }

    const decoder = new TextDecoder();
    let buffer = "";

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.trim() === "") continue;
          if (line.startsWith("data: ")) {
            const data = line.slice(6);
            if (data === "[DONE]") {
              return;
            }
            try {
              const json = JSON.parse(data);
              const content = json.choices?.[0]?.delta?.content;
              if (content) {
                yield content;
              }
            } catch (e) {
              // Skip invalid JSON chunks
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }
}

class ChatHistory {
  private messages: Message[] = [];

  addMessage(role: "user" | "assistant", content: string) {
    this.messages.push({ role, content });
  }

  getMessages(): Message[] {
    return [...this.messages];
  }

  clear() {
    this.messages = [];
  }

  getHistoryLength(): number {
    return this.messages.length;
  }
}

class CLI {
  private client: FireworksClient;
  private history: ChatHistory;
  private rl: readline.Interface;

  constructor() {
    this.client = new FireworksClient(FIREWORKS_API_KEY);
    this.history = new ChatHistory();
    this.rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });
  }

  private async getUserInput(): Promise<string> {
    return new Promise((resolve) => {
      this.rl.question(`${colors.cyan}You: ${colors.reset}`, (answer) => {
        resolve(answer.trim());
      });
    });
  }

  private async streamResponse(): Promise<void> {
    // Pause readline to prevent user input during streaming
    this.rl.pause();
    
    process.stdout.write(`\n${colors.green}AI: ${colors.reset}`);
    let fullResponse = "";

    try {
      for await (const chunk of this.client.streamChat(this.history.getMessages())) {
        process.stdout.write(chunk);
        fullResponse += chunk;
      }
      console.log("\n"); // New line after response
      this.history.addMessage("assistant", fullResponse);
    } catch (error) {
      console.error(`\n${colors.red}Error: ${error instanceof Error ? error.message : String(error)}${colors.reset}`);
    } finally {
      // Resume readline after streaming completes
      this.rl.resume();
    }
  }

  private handleCommand(input: string): boolean {
    const trimmed = input.trim();
    
    if (trimmed === "/exit" || trimmed === "/quit") {
      console.log(`${colors.yellow}Goodbye!${colors.reset}`);
      return true; // Signal to exit
    }

    if (trimmed === "/clear") {
      this.history.clear();
      console.log(`${colors.dim}Conversation history cleared.${colors.reset}`);
      return false; // Continue
    }

    if (trimmed === "/help") {
      console.log(`${colors.dim}Commands:`);
      console.log(`  /clear  - Clear conversation history`);
      console.log(`  /exit   - Exit the chatbot`);
      console.log(`  /help   - Show this help message${colors.reset}`);
      return false; // Continue
    }

    return false; // Not a command, continue with normal flow
  }

  close() {
    this.rl.close();
  }

  async run() {
    console.log(`${colors.green}CLI AI Chatbot${colors.reset}`);
    console.log(`${colors.dim}Type your message or /help for commands${colors.reset}\n`);

    while (true) {
      const input = await this.getUserInput();
      
      if (!input) {
        continue;
      }

      const shouldExit = this.handleCommand(input);
      if (shouldExit) {
        this.close();
        break;
      }

      // Skip empty commands
      if (input.startsWith("/")) {
        continue;
      }

      // Add user message to history
      this.history.addMessage("user", input);

      // Stream AI response
      await this.streamResponse();
    }
  }
}

// Start the CLI
const cli = new CLI();

// Handle Ctrl+C gracefully
process.on("SIGINT", () => {
  console.log(`\n${colors.yellow}Exiting...${colors.reset}`);
  cli.close();
  process.exit(0);
});

cli.run().catch((error) => {
  console.error(`${colors.red}Fatal error: ${error instanceof Error ? error.message : String(error)}${colors.reset}`);
  cli.close();
  process.exit(1);
});
