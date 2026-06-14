from boundary_smart_splitter import WordSplitter
from boundary_smart_splitter.v2_sentence import SentenceSplitter
from boundary_smart_splitter.v3_paragraph import ParagraphSplitter

text = """We're in another generational shift in software
Every twenty years or so, the way software gets built changes in a way that forces every engineering organization to restructure how they work. On-prem to cloud was the last one. Agents are the next one, and unlike the last shift, this one is happening in months, not years.

Every shift produces the same set of questions about who you depend on, how much of your business logic gets captive to a single vendor, and what it costs you to leave. The last time around, the enterprise mostly answered those questions late. The bill for that, in lock-in, unpredictable pricing, and failed migrations, got paid over the following decade.

We're about to do the exact same thing with models. We just lived through the last one. The lesson is right there.

The lesson from the cloud era
I spent years at HashiCorp in the middle of the cloud era. The single most important thing I learned there was that the surface story about cloud lock-in (outages, pricing power, failover) was the easy part of the argument. The deeper story was about what the hyperscalers were actually selling.

They were selling commodities. Storage, network, compute. AWS storage and GCP storage are, at the bytes-on-disk level, indistinguishable products. And because the underlying product is a commodity, the only durable way to keep customers consuming yours is to lock them in at the tooling layer. CloudFormation for AWS, ARM templates for Azure, Vertex for GCP. None of those tools had any intrinsic motivation to support a competitor's best-in-class feature at parity. Doing so would only have made it easier to leave.

Terraform's whole reason to exist was that this tooling-layer lock-in was real, expensive, and getting worse, and that the right answer was a neutral abstraction one layer up: the right to switch, and the ability to mix providers in a single deployment without rewriting your infrastructure.

That argument won. Not all at once, but it won. The enterprises that adopted neutrality early could push back on hyperscaler pricing because leaving was credible, and they could ride out outages because failover wasn't theoretical. Neither of those is available from a single-vendor stack.

The foundation labs are running the same play
Now look at what's happening with models.

The labs are selling you tokens. Tokens are a commodity, and increasingly so. The gap between frontier models has been closing, open-weight models are catching up fast, and the price-per-million-tokens chart has been on a steady downward slope for two years. The labs know this. They can see the same chart you can.

So their next move is to capture you at the harness, and you can watch them all making it at once. Claude Agent SDK, OpenAI's Agents API, Vertex AI Agent Builder. All the same shape. If they own the orchestration layer your business logic lives in, you keep consuming their tokens even when a better, cheaper, or more appropriate model exists somewhere else.

They have no commercial incentive to make their harness a great experience for running competitors' models alongside their own. CloudFormation never had a reason to provision GCP resources at parity. Claude Agent SDK has no reason to make calling GPT, Gemini, or Llama feel first-class. From the lab's perspective, making competitors' models work well inside their harness costs revenue. They aren't going to.

The harness lock-in is going to be harder to unwind than the model lock-in itself, because the harness is where your business logic lives.

Why model neutrality matters more than cloud neutrality did
If the lesson is the same, the response is the same: a neutral harness. The kind of layer Terraform was for cloud.

Model neutrality matters more than cloud neutrality ever did. Three reasons:

The rate of change is fundamentally different. You don't move applications from AWS to GCP month over month. You do it at contract renewal, or you do it in an outage. Maybe once every few years. But the labs are leapfrogging each other every quarter, often every month. A team locked to one provider isn't just exposed to outages and pricing. They're locked out of the next leap forward, every single time it happens.

Models are selectively commoditizing. The labs are racing for the harness layer precisely because raw model differentiation is eroding in the easy dimensions like basic reasoning, generic Q&A, and summarization. But they aren't commoditizing everywhere. Anthropic is currently the model to reach for on coding, though OpenAI is closing the gap, and OpenAI is ahead on multimodal. The rankings shift every few months. The right answer in a real production agent is often to use more than one model in the same workflow, routing each task to whichever is best at it today. That is only possible with a harness that doesn't take sides.

Open-weight models are a real option. Mistral, DeepSeek, Qwen. Self-hosting is credible in a way that "running your own private cloud" never really was for most enterprises. So neutrality isn't only defensive. It lets you mix closed and open models in the same agent, route to whichever is cheapest, fastest, or most accurate for the task, and switch the moment a better one ships.

There's another layer in the model era that didn't exist with cloud. Cloud neutrality was something you cashed in at contract renewal or during an outage. Model neutrality is something you exercise during a single agent run: choosing Claude for a coding step and GPT for an image step, failing over mid-execution when one provider rate-limits, dropping to a cheaper model where the expensive one isn't justified. Cloud neutrality stopped at the contract. Agent neutrality has to follow the request.

What a neutral harness actually means
A neutral harness is three things.

Open source. You can read every line of code. Nothing is hidden, silently captured, or positioned to optimize for a vendor at your expense. Closed-source agent frameworks shipped by a model lab are not neutral, no matter what the marketing claims.

Multi-model out of the box. Same harness, any backend. GPT, Claude, Gemini, Llama, Mistral, DeepSeek, Qwen, and anything you self-host behind your firewall. One agent definition, every provider first-class, because no provider owns the abstraction.

Profile-aware, not lowest-common-denominator. Neutrality is not the obligation to pretend every model is interchangeable. Every frontier model has its own personality, with strengths, prompt patterns, and tool-calling styles that don't generalize. A good neutral harness exposes model profiles, so you can exploit what makes each model great without being captive to any of them. The right to switch. Not the requirement to flatten.

This is what we built with Deep Agents, and what LangChain has always been at its core. It's open, multi-model, and profile-aware. A harness designed to outlast any single model provider.

We've done this before
Hyperscalers sold commodities and locked you in at the tooling layer. Labs are selling commodities and trying to lock you in at the harness. Terraform won the cloud era by being the neutral layer one step up. A neutral, open harness is the equivalent answer for the model era.

The difference is that this shift is moving an order of magnitude faster than the last one, and the cost of getting it wrong compounds an order of magnitude faster too."""

splitter = ParagraphSplitter(chunk_size=5, tolerance=50)
chunks = splitter.split(text)
print(len(chunks))

for i, chunk in enumerate(chunks, 1):
    print(f"--- Chunk {i} ({len(chunk)} chars) ---")
    print(chunk)
    print()
