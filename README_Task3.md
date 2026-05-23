# 🤖 Task 5: Auto Tagging Support Tickets Using LLM
**DevelopersHub Corporation – AI/ML Engineering Advanced Internship**

## Objective
Automatically tag support tickets into categories using a Large Language Model (LLM) with prompt engineering. Compare zero-shot vs few-shot performance and output top-3 most probable tags per ticket.

## Methodology / Approach
- **Dataset:** 56 support tickets across 7 categories (Billing, Technical, Account, Shipping, Feature Request, Refund, General Inquiry)
- **Zero-Shot:** Prompt contains only the task description + category list — no examples
- **Few-Shot (3-Shot):** Prompt includes 3 labeled example tickets (one per representative category) before the query
- **Output:** Top-3 ranked tags per ticket with confidence scores
- **LLM Integration:** Architecture uses structured prompt templates (JSON output format). Connects to Anthropic Claude / OpenAI via simple API swap

## Key Results / Observations
| Method | Accuracy | F1 (weighted) | Top-3 Inclusion Rate |
|---|---|---|---|
| Zero-Shot | ~0.71 | ~0.70 | ~0.92 |
| **Few-Shot** | **~0.77** | **~0.76** | **~0.95** |

- Few-shot learning boosts accuracy by ~6% by providing label-conditioning context
- "Billing & Payment" and "Technical Issue" are easiest (distinctive vocabulary)
- "General Inquiry" has lowest F1 — overlaps semantically with other categories
- Top-3 inclusion rate (~95%) makes the system highly reliable for routing

## Files
| File | Description |
|---|---|
| `Task5_Auto_Tagging_LLM.ipynb` | Full solution notebook |
| `ticket_distribution.png` | Category distribution chart |
| `llm_tagging_evaluation.png` | Zero-shot vs few-shot comparison + confusion matrix |

## Setup
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
# Optional for real LLM: pip install anthropic openai
jupyter notebook Task5_Auto_Tagging_LLM.ipynb
```

## Connecting a Real LLM API
```python
import anthropic, json

client = anthropic.Anthropic(api_key="YOUR_API_KEY")

def classify_with_claude(ticket_text, few_shot_examples):
    prompt = build_few_shot_prompt(ticket_text, few_shot_examples)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    return json.loads(message.content[0].text)['tags']
```

## Skills Demonstrated
Prompt engineering · LLM-based text classification · Zero-shot and few-shot learning · Multi-class prediction and ranking
