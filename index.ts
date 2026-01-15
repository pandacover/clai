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

// Get API keys from environment
const API_KEY = process.env.FIREWORKS_API_KEY;
if (!API_KEY || API_KEY === "your-api-key-here") {
  console.error(`${colors.red}Error: FIREWORKS_API_KEY not set in .env file${colors.reset}`);
  console.error(`${colors.dim}Please set FIREWORKS_API_KEY=your-actual-key in .env${colors.reset}`);
  process.exit(1);
}

const GOOGLE_SEARCH_API_KEY = process.env.GOOGLE_SEARCH_API_KEY;
if (!GOOGLE_SEARCH_API_KEY) {
  console.error(`${colors.red}Error: GOOGLE_SEARCH_API_KEY not set in .env file${colors.reset}`);
  console.error(`${colors.dim}Please set GOOGLE_SEARCH_API_KEY=your-actual-key in .env${colors.reset}`);
  process.exit(1);
}

// TypeScript assertion: API_KEY is guaranteed to be a string after the check above
const FIREWORKS_API_KEY: string = API_KEY;

const API_URL = "https://api.fireworks.ai/inference/v1/chat/completions";
const GOOGLE_SEARCH_URL = "https://www.googleapis.com/customsearch/v1";

interface Message {
  role: "user" | "assistant" | "system" | "tool";
  content: string | null;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
  name?: string;
}

interface ToolCall {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;
  };
}

interface ToolDefinition {
  type: "function";
  function: {
    name: string;
    description: string;
    parameters: {
      type: "object";
      properties: Record<string, { type: string; description: string }>;
      required: string[];
    };
  };
}

interface GoogleSearchResult {
  title: string;
  link: string;
  snippet: string;
}

// Web search tool definition
const WEB_SEARCH_TOOL: ToolDefinition = {
  type: "function",
  function: {
    name: "web_search",
    description: "Search the internet for current information",
    parameters: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "Search query",
        },
      },
      required: ["query"],
    },
  },
};

// Web search implementation
async function webSearch(query: string): Promise<string> {
  try {
    // Google Custom Search API requires both API key and CX (Custom Search Engine ID)
    // For now, we'll use the API key. If CX is needed, it should be added to .env
    const cx = process.env.GOOGLE_SEARCH_CX || "";
    
    const url = new URL(GOOGLE_SEARCH_URL);
    // GOOGLE_SEARCH_API_KEY is guaranteed to be a string after the check above
    url.searchParams.set("key", GOOGLE_SEARCH_API_KEY as string);
    url.searchParams.set("q", query);
    if (cx) {
      url.searchParams.set("cx", cx);
    }
    url.searchParams.set("num", "10"); // Get up to 10 results

    const response = await fetch(url.toString());
    
    if (!response.ok) {
      const error = await response.text();
      throw new Error(`Google Search API Error: ${response.status} ${response.statusText}\n${error}`);
    }

    const data = await response.json() as { items?: Array<{ title?: string; link?: string; snippet?: string }> };
    
    if (!data.items || data.items.length === 0) {
      return "No search results found.";
    }

    const results: GoogleSearchResult[] = data.items.map((item) => ({
      title: item.title || "",
      link: item.link || "",
      snippet: item.snippet || "",
    }));

    // Format results as a readable string
    let formattedResults = `Search results for "${query}":\n\n`;
    results.forEach((result, index) => {
      formattedResults += `${index + 1}. ${result.title}\n`;
      formattedResults += `   URL: ${result.link}\n`;
      formattedResults += `   ${result.snippet}\n\n`;
    });

    return formattedResults;
  } catch (error) {
    return `Error performing web search: ${error instanceof Error ? error.message : String(error)}`;
  }
}

class FireworksClient {
  private apiKey: string;
  private apiUrl: string;
  private tools: ToolDefinition[];

  constructor(apiKey: string, tools: ToolDefinition[] = []) {
    this.apiKey = apiKey;
    this.apiUrl = API_URL;
    this.tools = tools;
  }

  async *streamChat(messages: Message[]): AsyncGenerator<string | { toolCalls: ToolCall[] }, void, unknown> {
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
        tools: this.tools.length > 0 ? this.tools : undefined,
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
    const accumulatedToolCalls: Map<number, Partial<ToolCall>> = new Map();

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
              // Yield any remaining accumulated tool calls
              if (accumulatedToolCalls.size > 0) {
                const completeToolCalls: ToolCall[] = Array.from(accumulatedToolCalls.values())
                  .filter((tc): tc is ToolCall => 
                    tc.id !== undefined && 
                    tc.function !== undefined && 
                    tc.function.name !== undefined && 
                    tc.function.arguments !== undefined
                  )
                  .map(tc => ({
                    id: tc.id!,
                    type: "function" as const,
                    function: {
                      name: tc.function!.name!,
                      arguments: tc.function!.arguments!,
                    },
                  }));
                if (completeToolCalls.length > 0) {
                  yield { toolCalls: completeToolCalls };
                }
              }
              return;
            }
            try {
              const json = JSON.parse(data);
              const delta = json.choices?.[0]?.delta;
              
              // Check for tool calls (streaming format)
              if (delta?.tool_calls) {
                for (const toolCallDelta of delta.tool_calls) {
                  const index = toolCallDelta.index;
                  if (index !== undefined) {
                    if (!accumulatedToolCalls.has(index)) {
                      accumulatedToolCalls.set(index, { id: toolCallDelta.id, function: { name: "", arguments: "" } });
                    }
                    const toolCall = accumulatedToolCalls.get(index)!;
                    if (toolCallDelta.id) {
                      toolCall.id = toolCallDelta.id;
                    }
                    if (toolCallDelta.function) {
                      if (toolCallDelta.function.name) {
                        toolCall.function = { 
                          name: toolCallDelta.function.name, 
                          arguments: toolCall.function?.arguments || "" 
                        };
                      }
                      if (toolCallDelta.function.arguments) {
                        toolCall.function = { 
                          name: toolCall.function?.name || "", 
                          arguments: (toolCall.function?.arguments || "") + toolCallDelta.function.arguments 
                        };
                      }
                    }
                  }
                }
              }
              
              // Check for content
              const content = delta?.content;
              if (content) {
                yield content;
              }
            } catch (e) {
              // Skip invalid JSON chunks
            }
          }
        }
      }
      
      // Yield any accumulated tool calls at the end
      if (accumulatedToolCalls.size > 0) {
        const completeToolCalls: ToolCall[] = Array.from(accumulatedToolCalls.values())
          .filter((tc): tc is ToolCall => 
            tc.id !== undefined && 
            tc.function !== undefined && 
            tc.function.name !== undefined && 
            tc.function.arguments !== undefined
          )
          .map(tc => ({
            id: tc.id!,
            type: "function" as const,
            function: {
              name: tc.function!.name!,
              arguments: tc.function!.arguments!,
            },
          }));
        if (completeToolCalls.length > 0) {
          yield { toolCalls: completeToolCalls };
        }
      }
    } finally {
      reader.releaseLock();
    }
  }
}

class ChatHistory {
  private messages: Message[] = [];

  addMessage(role: "user" | "assistant" | "tool", content: string | null, toolCalls?: ToolCall[], toolCallId?: string, name?: string) {
    const message: Message = { role, content };
    if (toolCalls) {
      message.tool_calls = toolCalls;
    }
    if (toolCallId) {
      message.tool_call_id = toolCallId;
    }
    if (name) {
      message.name = name;
    }
    this.messages.push(message);
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
    this.client = new FireworksClient(FIREWORKS_API_KEY, [WEB_SEARCH_TOOL]);
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
    let toolCalls: ToolCall[] = [];
    let hasToolCalls = false;

    try {
      for await (const chunk of this.client.streamChat(this.history.getMessages())) {
        if (typeof chunk === "string") {
          process.stdout.write(chunk);
          fullResponse += chunk;
        } else if (chunk && typeof chunk === "object" && "toolCalls" in chunk) {
          // Collect tool calls (may come at the end of the stream)
          toolCalls = chunk.toolCalls;
          hasToolCalls = true;
        }
      }
      
      // If there are tool calls, execute them
      if (hasToolCalls && toolCalls.length > 0) {
        console.log("\n"); // New line before tool execution
        process.stdout.write(`${colors.dim}[Executing tools...]${colors.reset}\n`);
        
        // IMPORTANT: Add assistant message with tool_calls FIRST
        // The API requires the assistant message with tool_calls before tool responses
        this.history.addMessage("assistant", fullResponse || null, toolCalls);
        
        // Execute all tool calls and add results
        for (const toolCall of toolCalls) {
          const functionName = toolCall.function.name;
          let args: any;
          try {
            args = JSON.parse(toolCall.function.arguments);
          } catch (e) {
            args = { query: toolCall.function.arguments }; // Fallback for malformed JSON
          }
          
          let result: string;
          
          if (functionName === "web_search") {
            result = await webSearch(args.query);
          } else {
            result = `Unknown tool: ${functionName}`;
          }
          
          // Add tool result to history (after assistant message with tool_calls)
          this.history.addMessage("tool", result, undefined, toolCall.id, functionName);
        }
        
        // Get response after tool execution
        process.stdout.write(`\n${colors.green}AI: ${colors.reset}`);
        fullResponse = "";
        toolCalls = [];
        hasToolCalls = false;
        
        // Stream the follow-up response (recursive call for multiple tool rounds)
        await this.streamResponse();
        return;
      }
      
      console.log("\n"); // New line after response
      this.history.addMessage("assistant", fullResponse || null, toolCalls.length > 0 ? toolCalls : undefined);
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
