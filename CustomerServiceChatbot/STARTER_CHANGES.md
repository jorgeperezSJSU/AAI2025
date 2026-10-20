# How this adapts the provided starter code

Source: https://github.com/myrah/build-ai-agents-and-chatbots-with-langgraph-2021112
Course: Build AI Agents and Chatbots with LangGraph (LinkedIn Learning).
Original copyright and terms are retained in NOTICE and LICENSE in this directory.
This folder is an educational adaptation; the repository root license does not replace the starter's license here.

| Starter component | Adaptation |
|---|---|
| Chapter 06 `RouterAgent`, conditional edges, `agent_node`, `functools.partial` | Retained that structure; added REFUND; typed route goes into state rather than appearing as an assistant message. |
| Chapter 04 `OrdersAgent.call_llm`, `is_tool_call`, `call_tools` | Generalized into `SpecialistAgent`, instantiated three times with separate prompts and tools. |
| Chapter 04 tool loop | Moved return outside loop so every tool call receives a result. |
| Chapter 03 product price tool | Retained name-prefix lookup; removed regex interpretation; added budget recommendations. |
| Chapter 03/04 `MemorySaver` and thread ID | One shared parent checkpointer retains history across all specialties; child graphs do not independently replay/checkpoint it. |
| AzureChatOpenAI and embedded configuration | Replaced with ChatGoogleGenerativeAI and GOOGLE_API_KEY from Colab Secrets/environment. |
| Laptop pricing/orders CSVs | Embedded the five original records from each CSV so the notebook is self-contained. Historical dates preserved; no fabricated shipment status. |
| PDF/Chroma product features | Simplified to prices and shipping times. This assignment does not require RAG; no unsupported specifications are invented. |
| Order-quantity mutation | Omitted; this support demo only reads order information. |
| `%run` of earlier notebooks | Replaced with organized self-contained cells to avoid running old demos during imports. |
| Scripted user-input loop | Expanded to six live turns, visible agent/tool names, memory checks and downloadable output evidence. |

Modified for this assignment with AI assistance. Review and understand the code before submission.
