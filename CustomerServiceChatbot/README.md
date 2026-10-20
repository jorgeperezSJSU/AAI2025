# TechHaven — Agentic Customer Service Chatbot

A Google Gemini + LangGraph adaptation of the instructor's provided starter notebooks.
Three separate tool-using agents handle laptop recommendations, order status/details,
and refund policy. A model-powered router selects the agent. `MemorySaver` keeps
conversation state across turns using the same `thread_id`.

[Open notebook in Google Colab](https://colab.research.google.com/github/jorgeperezSJSU/AAI2025/blob/main/CustomerServiceChatbot/Customer_Service_Chatbot.ipynb)

## Run in Colab

1. Open the notebook above and select **File → Save a copy in Drive** if needed.
2. In that notebook's Secrets panel (key icon), add/enable `GOOGLE_API_KEY`.
   Secrets are account-specific: check Notebook access for this notebook, too.
3. Select **Runtime → Run all**. The final optional interactive cell is disabled by default.
4. Review all six replies. The notebook saves and offers to download the actual output transcript.
5. Save/download the executed `.ipynb` with output cells included. Upload it and
   `successful_outputs.txt` to this GitHub folder, or send them back for review/upload.

**Live-run status:** The code was checked locally with offline model doubles. A live
Google AI run remains required in the user's Colab runtime, where the key is stored.
No historical starter outputs or fabricated responses count as this project's results.

## Assignment requirements

| Requirement | Implementation | Evidence to capture |
|---|---|---|
| OpenAI or Google AI API | `ChatGoogleGenerativeAI` | Live notebook responses |
| Role-defining system prompt | `SYSTEM_PROMPT` and three specialist prompts | Source cells |
| Three inquiry types using different agents | PRODUCT, ORDER, REFUND; separate agent instances and tools | Agent/tool labels in demo |
| At least three turns with MemorySaver | Router compiled with `MemorySaver()`; one thread for six turns | Remembered-turn counts + contextual answers |
| GitHub link and successful outputs | This folder + executed notebook/transcript | Submit after live run |

## Demo expectations (not generated responses)

1. Under $1500: GammaAir X ($1399) and AlphaBook Pro ($1499).
2. Fastest among those: AlphaBook Pro (2 shipping days vs. 7).
3. “How much does that one cost?”: $1499 without restating the model name.
4. Return after 10 days: within the fictional 30-day window, subject to conditions.
5. ORD-7311: two NanoEdge Flex laptops, historical date 2024-10-19; status not recorded.
6. “How many laptops are in that order?”: two, without restating the order ID.

Automatic checks verify routing and retained customer turns. Manually review answer
accuracy as well; saved-message counts alone do not prove semantic memory.

## Local run (Python 3.11+)

```bash
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set your key privately. Then:

```bash
python chatbot.py --demo
python chatbot.py
python -m unittest test_chatbot -v
```

The first command runs the demo; the second starts a terminal chat. Tests do not call
Google. All store data is fictional. `MemorySaver` lasts only while this process lives;
restarting Colab clears it. The demo makes multiple API calls and uses your API quota.
If you see 429, wait for quota recovery or check the project's limits. For a 404 model
error, replace `GEMINI_MODEL` with an available text model supporting tool calls and
structured output. No fallback silently changes models or enables billing.

## Source and design

See [STARTER_CHANGES.md](STARTER_CHANGES.md) for the exact starter components adapted.
The graph uses Router → selected specialist → END. Inside each specialist it uses
LLM → tools → LLM until a final answer is produced. Each agent has its own prompt and
tools, while sharing one underlying Gemini model and parent conversation history.
This is a classroom demo, not an authenticated production order-management service.
